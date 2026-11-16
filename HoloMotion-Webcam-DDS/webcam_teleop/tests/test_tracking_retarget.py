import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'webcam_teleop'))
from retarget import WebcamRetarget
from test_tracking_filter import landmarks


class TrackingRetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
        cls.robot = yaml.safe_load((ROOT / 'checkpoints/HoloMotion_motion_tracking_model_v2.0.0/config.yaml').read_text())['robot']

    def setUp(self):
        r = self.robot
        self.adapter = WebcamRetarget(self.model, r['dof_names'], r['body_names'],
            [r['init_state']['default_joint_angles'][n] for n in r['dof_names']])
        self.pose = landmarks()
        self.adapter.calibrate(self.pose, strict=False)

    def test_duplicate_camera_frame_never_reruns_ik_or_input_filter(self):
        a = self.adapter
        with patch.object(a, '_solve_target', wraps=a._solve_target) as solve:
            a.update(self.pose, sample_id=1, timestamp=0.)
            target = a._target_q.copy()
            for _ in range(10):
                a.update(self.pose, sample_id=1, timestamp=0.)
            self.assertEqual(solve.call_count, 1)
            np.testing.assert_array_equal(a._target_q, target)
            self.assertEqual(a.pose_filter.timestamp, 0.)
            a.update(self.pose, sample_id=2, timestamp=.05)
            self.assertEqual(solve.call_count, 2)

    def test_low_confidence_and_spikes_do_not_change_ik_target(self):
        a = self.adapter
        a.update(self.pose, sample_id=1, timestamp=0.)
        target = a._target_q.copy()
        p = self.pose.copy(); p[15, 3] = 0.
        a.update(p, sample_id=2, timestamp=.05)
        np.testing.assert_array_equal(a._target_q, target)
        p = self.pose.copy(); p[15, 2] += 1.
        a.update(p, sample_id=3, timestamp=.1)
        np.testing.assert_array_equal(a._target_q, target)

    def test_rate_limit_and_joint_bounds_survive_large_input_change(self):
        a = self.adapter
        a.update(self.pose, sample_id=1, timestamp=0.)
        before = a.q.copy()
        p = self.pose.copy(); p[13, :3] = [.46, -.5, -.03]; p[15, :3] = [.7, -.5, -.04]
        for i in range(1, 20):
            q, _, _ = a.update(p, sample_id=i + 1, timestamp=i * .05)
            self.assertLessEqual(float(np.max(np.abs(q - before))), .050001)
            for arm in a.arms.values():
                angles = q[arm['indices']]
                self.assertTrue(np.all(angles >= arm['bounds'][:, 0]))
                self.assertTrue(np.all(angles <= arm['bounds'][:, 1]))
            before = q.copy()
        self.assertTrue(np.isfinite(a.q).all())

    def test_calibration_clears_live_filter_and_hold_state(self):
        a = self.adapter
        a.update(self.pose, sample_id=8, timestamp=1.)
        a.calibrate(self.pose, strict=False)
        self.assertIsNone(a._sample_id)
        self.assertIsNone(a.pose_filter.timestamp)
        self.assertFalse(a.joint_hold.holding.any())


if __name__ == '__main__':
    unittest.main()
