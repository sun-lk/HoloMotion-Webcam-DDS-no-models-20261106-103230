"""Exercise HTTP gates and real IK without starting a server or publishing commands."""
import ast
import io
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'webcam_teleop'))
from retarget import WebcamRetarget

TREE = ast.parse((ROOT / 'webcam_teleop/server.py').read_text(encoding='utf-8'))
HANDLER = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'Handler')
POST = next(n for n in HANDLER.body if isinstance(n, ast.FunctionDef) and n.name == 'do_POST')
CAPTURE = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'capture_alignment_target')


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.state = dict(ready=True, calibrated=False, fallen=False, running=False,
            robot_enabled=False, auto_calibrated=False, alignment_status='idle',
            align=False, cancel_alignment=False)
        self.scope = dict(np=np, json=json, time=time, LOCK=threading.Lock(), STATE=self.state, BACKEND='g1ra')
        exec(compile(ast.Module(body=[POST], type_ignores=[]), 'server.py', 'exec'), self.scope)

    def call(self, action, **values):
        data = json.dumps(dict(action=action, **values)).encode()
        replies = []
        handler = SimpleNamespace(headers={'Content-Length': str(len(data))},
            server=SimpleNamespace(server_port=8765), path='/api/control', rfile=io.BytesIO(data),
            respond=lambda body, status=200: replies.append((status, body)))
        self.scope['do_POST'](handler)
        return replies[-1][0]

    def test_align_is_explicit_request_and_does_not_enable_policy(self):
        self.assertEqual(self.call('align', move_robot=True), 200)
        self.assertTrue(self.state['align'])
        self.assertFalse(self.state['running'])
        self.assertFalse(self.state['robot_enabled'])

    def test_display_switch_does_not_change_motion_or_feedback(self):
        self.state.update(robot_view_supported=True, feedback_source='robot',
                          running=True, robot_enabled=True)
        self.assertEqual(self.call('set_display_source', source='simulation'), 200)
        self.assertEqual(self.state['display_source'], 'simulation')
        self.assertTrue(self.state['running'])
        self.assertTrue(self.state['robot_enabled'])
        self.assertEqual(self.state['feedback_source'], 'robot')
        self.assertEqual(self.call('set_display_source', source='robot'), 200)
        self.assertEqual(self.call('set_display_source', source='unknown'), 400)

    def test_feedback_switch_requires_pause_and_cancels_takeover(self):
        self.state.update(running=True, calibrated=True, robot_enabled=True,
                          feedback_source='robot', display_source='robot')
        self.assertEqual(self.call('set_feedback_source', source='simulation'), 400)
        self.state['running'] = False
        self.assertEqual(self.call('set_feedback_source', source='simulation'), 200)
        self.assertEqual(self.state['feedback_source'], 'simulation')
        self.assertEqual(self.state['display_source'], 'robot')
        self.assertTrue(self.state['feedback_pending'])
        self.assertFalse(self.state['robot_enabled'])
        self.assertFalse(self.state['calibrated'])
        self.assertEqual(self.call('start'), 400)
        self.assertEqual(self.call('align', move_robot=True), 400)

    def test_feedback_switch_rejects_alignment_and_non_robot_backend(self):
        self.state.update(alignment_status='moving')
        self.assertEqual(self.call('set_feedback_source', source='simulation'), 400)
        self.state.update(alignment_status='idle')
        self.scope['BACKEND'] = 'direct'
        self.assertEqual(self.call('set_feedback_source', source='simulation'), 400)

    def test_old_calibrate_and_unlabelled_request_cannot_move_robot(self):
        self.assertEqual(self.call('calibrate'), 400)
        self.assertEqual(self.call('align'), 400)
        self.assertFalse(self.state['align'])

    def test_start_does_not_require_actual_arrival(self):
        self.state.update(calibrated=True, robot_enabled=True, alignment_status='moving')
        self.assertEqual(self.call('start'), 200)
        self.state['alignment_status'] = 'holding'
        self.assertEqual(self.call('start'), 200)
        self.state['fallen'] = True
        self.assertEqual(self.call('start'), 400)

    def test_pause_off_and_camera_change_cancel_alignment(self):
        for action, values in [('pause', {}), ('set_robot', {'enabled': False}), ('new_camera', {})]:
            self.state.update(alignment_status='moving', align=True, calibrated=False, robot_enabled=True)
            self.assertEqual(self.call(action, **values), 200)
            self.assertTrue(self.state['cancel_alignment'])
            self.assertFalse(self.state['robot_enabled'])
            self.assertFalse(self.state['align'])

    def test_simulation_cannot_request_real_alignment(self):
        self.scope['BACKEND'] = 'direct'
        self.assertEqual(self.call('align'), 400)

    def test_pure_mujoco_start_needs_no_robot_alignment_or_feedback(self):
        self.scope['BACKEND'] = 'direct'
        self.state.update(calibrated=True, auto_calibrated=True, robot_enabled=False,
                          alignment_status='idle')
        self.assertEqual(self.call('start'), 200)
        self.assertTrue(self.state['running'])
        self.assertFalse(self.state['robot_enabled'])
        self.assertEqual(self.call('align', move_robot=True), 400)

    def test_pause_cancels_alignment_before_worker_starts(self):
        self.call('align', move_robot=True)
        self.assertEqual(self.state['alignment_status'], 'idle')
        self.call('pause')
        self.assertFalse(self.state['align'])
        self.assertTrue(self.state['cancel_alignment'])

    def test_old_timestamps_cannot_refresh_pose_freshness(self):
        self.state.update(sequence=-1, pose_source_clock=False, pose_time=0.)

        def send(seq, stamp):
            data = json.dumps(dict(sequence=seq, timestamp_ms=stamp,
                                  landmarks=np.ones((33, 4)).tolist())).encode()
            replies = []
            handler = SimpleNamespace(headers={'Content-Length': str(len(data))},
                server=SimpleNamespace(server_port=8765), path='/api/pose', rfile=io.BytesIO(data),
                respond=lambda body, status=200: replies.append(status))
            self.scope['do_POST'](handler)
            return replies[-1]

        self.assertEqual(send(0, 100.), 200)
        received = self.state['last_pose']
        self.assertEqual(send(1, 100.), 400)
        self.assertEqual(self.state['last_pose'], received)
        self.assertEqual(send(1, 150.), 200)
        self.assertAlmostEqual(self.state['pose_time'], .15)
        self.call('new_camera')
        self.assertEqual(self.state['last_pose'], 0.)
        self.assertEqual(send(0, 1.), 200)

    def test_real_ik_converges_for_frozen_sample_without_io(self):
        scope = {'np': np}
        exec(compile(ast.Module(body=[CAPTURE], type_ignores=[]), 'server.py', 'exec'), scope)
        robot = yaml.safe_load((ROOT / 'checkpoints/HoloMotion_motion_tracking_model_v2.0.0/config.yaml').read_text())['robot']
        m = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
        names = robot['dof_names']
        defaults = np.array([robot['init_state']['default_joint_angles'][n] for n in names])
        adapter = WebcamRetarget(m, names, robot['body_names'], defaults)
        p = np.zeros((33, 4)); p[:, 3] = .99
        for i, xyz in {11:[.2,-.5,0],12:[-.2,-.5,0],23:[.13,0,0],24:[-.13,0,0],
                       13:[.22,-.23,0],15:[.22,.02,-.01],14:[-.22,-.23,0],16:[-.22,.02,-.01]}.items():
            p[i, :3] = xyz
        before = p.copy()
        target = scope['capture_alignment_target'](adapter, SimpleNamespace(names=names), p)
        self.assertEqual(target.shape, (29,))
        self.assertTrue(np.isfinite(target).all())
        np.testing.assert_array_equal(p, before)
        again = scope['capture_alignment_target'](adapter, SimpleNamespace(names=names), p)
        np.testing.assert_allclose(again, target, atol=1e-8)
        # A raised arm is accepted without requiring the operator to mimic nominal stance.
        p[13, :3] = [.46, -.5, -.03]
        p[15, :3] = [.70, -.5, -.04]
        raised = scope['capture_alignment_target'](adapter, SimpleNamespace(names=names), p)
        self.assertTrue(np.isfinite(raised).all())
        self.assertGreater(float(np.max(np.abs(raised - target))), .1)


if __name__ == '__main__':
    unittest.main()
