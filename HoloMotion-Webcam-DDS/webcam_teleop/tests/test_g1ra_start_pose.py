"""No sockets or robot commands: exercise takeover with synthetic state frames."""
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1ra_backend import G1raBackend
from parallel_simulation import ParallelSimulation


class StartPoseTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        m = mujoco.MjModel.from_xml_path(str(root / 'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
        d = mujoco.MjData(m)
        names = [m.joint(int(m.actuator_trnid[i, 0])).name for i in range(m.nu)]
        self.q = np.linspace(-.1, .1, len(names)).astype(np.float32)
        self.sent = []
        ev = SimpleNamespace(
            m=m, d=d, obs_builder=SimpleNamespace(reset=lambda: None),
            dof_names_onnx=names[::-1], default_angles_onnx=np.zeros(29),
            action_scale_onnx=np.ones(29), counter=99, policy_kv_cache=np.ones(4),
        )

        def reset_policy():
            ev.counter = 0
            ev.policy_kv_cache.fill(0)
            mujoco.mj_resetData(m, d)
            d.qvel[:] = 9.0  # Detect stale simulation velocity after reset.

        ev.reset_state_teleport = reset_policy
        backend = G1raBackend.__new__(G1raBackend)
        backend.ev = ev
        backend.protocol = SimpleNamespace(N_JOINTS=29, ROBOT_STATE_POLICY=3, ROBOT_STATE_POLICY_HOLD=4,
                                           PHASE_HOLD=3, PHASE_ALIGNMENT=1, PHASE_POLICY=2)
        backend.control_mode = 'full_body'
        backend.names = names
        backend.qpos_idx = [int(m.joint(n).qposadr[0]) for n in names]
        backend.qvel_idx = [int(m.joint(n).dofadr[0]) for n in names]
        backend._state = SimpleNamespace(q=self.q, dq=np.full(29, .03),
            quat_wxyz=np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)]),
            gyro=np.array([.1, -.2, .3]))
        backend._state_time = time.monotonic()
        backend.state_timeout_s = .5
        backend._state_is_valid = True
        backend.poll_link = lambda: None
        self.phases = []
        def capture(q, phase=0):
            self.sent.append(q.copy())
            self.phases.append(phase)
        backend._publish = capture
        backend.pairs = 0
        backend.needs_sync = True
        backend._last_command = backend._reference_q = None
        backend._stale_warned = False
        backend.lock_lower_body = True
        backend.lower_idx = [i for i, n in enumerate(names) if any(k in n for k in ('hip_', 'knee', 'ankle'))]
        backend.stance = np.zeros(29)
        self.backend = backend

    def test_takeover_uses_measured_pose_and_clears_policy_history(self):
        b = self.backend
        self.assertTrue(b.sync_hold())
        self.assertEqual(self.phases, [3])
        np.testing.assert_array_equal(self.sent[0], self.q)
        np.testing.assert_array_equal(b.ev.d.qpos[b.qpos_idx], self.q)
        np.testing.assert_array_equal(b.ev.d.qvel[:3], np.zeros(3))
        np.testing.assert_array_equal(b.ev.d.qvel[3:6], b._state.gyro)
        np.testing.assert_array_equal(b.ev.d.qvel[b.qvel_idx], b._state.dq)
        np.testing.assert_array_equal(b.ev.actions_onnx, self.q[::-1])
        self.assertEqual(b.ev.counter, 0)
        np.testing.assert_array_equal(b.ev.policy_kv_cache, np.zeros(4))

    def test_stale_state_never_sends_default_pose(self):
        b = self.backend
        b._state_time -= 1.
        self.assertFalse(b.sync_hold())
        self.assertEqual(self.sent, [])
        self.assertEqual(b.ev.counter, 99)

    def test_invalid_state_is_rejected(self):
        self.backend._state.gyro[0] = np.nan
        self.assertFalse(self.backend.sync_hold())
        self.assertEqual(self.sent, [])

    def test_reference_starts_at_actual_pose_and_locked_legs_stay_there(self):
        b = self.backend
        b.sync_hold()
        target = self.q + 1.
        np.testing.assert_array_equal(b.reference_target(target, .05), self.q)
        next_q = b.reference_target(target, .05)
        self.assertLessEqual(float(np.max(np.abs(next_q - self.q))), .050001)
        np.testing.assert_array_equal(next_q[b.lower_idx], self.q[b.lower_idx])

    def test_policy_reads_new_measurement_and_output_is_slew_limited(self):
        b = self.backend
        b.sync_hold()
        b._state.q = self.q + .01
        self.assertTrue(b.prepare_policy_step())
        np.testing.assert_allclose(b.ev.d.qpos[b.qpos_idx], b._state.q)
        b.ev.target_dof_pos_by_name = {n: float(q + 1.) for n, q in zip(b.names, self.q)}
        b.step()
        self.assertEqual(self.phases[-1], 2)
        self.assertLessEqual(float(np.max(np.abs(self.sent[-1] - self.q))), .020001)
        np.testing.assert_array_equal(self.sent[-1][b.lower_idx], self.q[b.lower_idx])

    def test_stale_feedback_stops_until_resynchronized(self):
        b = self.backend
        b.sync_hold()
        b._state_time -= 1.
        self.assertFalse(b.prepare_policy_step())
        count = len(self.sent)
        b.step()
        self.assertEqual(len(self.sent), count)
        b._state_time = time.monotonic()
        self.assertFalse(b.prepare_policy_step())
        b._state.q = self.q + .02
        self.assertTrue(b.sync_hold())
        np.testing.assert_array_equal(self.sent[-1], b._state.q)

    def test_simulated_feedback_is_seeded_once_and_not_overwritten_by_real_state(self):
        b = self.backend
        b.feedback_source = 'simulation'
        b.simulation = ParallelSimulation(b.ev, lambda **kw: None)
        self.assertTrue(b.sync_hold())
        self.assertIs(b.ev.d, b.simulation.data)
        np.testing.assert_allclose(b.ev.d.qpos[b.qpos_idx], self.q)
        b.ev.d.qpos[b.qpos_idx] = self.q + .03
        b._state.q = self.q + .5
        self.assertTrue(b.prepare_policy_step())
        np.testing.assert_allclose(b.ev.d.qpos[b.qpos_idx], self.q + .03)
        self.assertTrue(b.prepare_policy_step(force_measured=True))
        self.assertIs(b.ev.d, b.simulation.measured_data)
        np.testing.assert_allclose(b.ev.d.qpos[b.qpos_idx], self.q + .5)
        b._state_time -= 1
        self.assertFalse(b.prepare_policy_step())

    def test_measured_render_snapshot_does_not_change_simulated_feedback(self):
        b = self.backend
        b.feedback_source = 'simulation'
        b.simulation = ParallelSimulation(b.ev, lambda **kw: None)
        b.sync_hold()
        before = b.ev.d.qpos.copy()
        b._state.q = self.q + .2
        snap = b.measured_snapshot(np.zeros((30, 3)))
        np.testing.assert_array_equal(b.ev.d.qpos, before)
        np.testing.assert_allclose(snap[0][b.qpos_idx], self.q + .2)

    def test_simulated_feedback_last_action_matches_applied_arm_targets(self):
        b = self.backend
        b.feedback_source = 'simulation'
        b.simulation = ParallelSimulation(b.ev, lambda **kw: None)
        b.sync_hold()
        b.ev.target_dof_pos_by_name = {n: float(q+1) for n,q in zip(b.names,self.q)}
        b.step()
        targets = b.simulation.last_targets
        np.testing.assert_allclose(b.ev.actions_onnx,
                                   [targets[n] for n in b.ev.dof_names_onnx], atol=1e-7)
        for j,n in enumerate(b.names):
            if any(k in n for k in ('shoulder','elbow','wrist')):
                self.assertAlmostEqual(targets[n], self.sent[-1][j])

    def test_reset_does_not_publish(self):
        self.backend.reset()
        self.assertTrue(self.backend.needs_sync)
        self.assertEqual(self.sent, [])

    def test_alignment_sender_rejects_stale_state_and_large_steps(self):
        b = self.backend
        b.sync_hold()
        b.send_alignment(self.q + .004)
        self.assertEqual(self.phases[-1], 1)
        count = len(self.sent)
        with self.assertRaises(ValueError):
            b.send_alignment(self.q + .1)
        self.assertEqual(len(self.sent), count)
        b._state_time -= 1.
        with self.assertRaises(ValueError):
            b.send_alignment(self.q + .005)
        self.assertEqual(len(self.sent), count)

    def test_arm_sdk_hold_is_ready_without_whole_body_policy(self):
        b = self.backend
        b.control_mode = 'arm_sdk'
        for state in (3, 4, 3, 4):
            b.robot_state = state
            self.assertTrue(b.alignment_control_ready())
        for state in (0, 1, 2, 5):
            b.robot_state = state
            with self.assertRaises(ValueError):
                b.alignment_control_ready()

    def test_full_body_hold_still_requires_policy(self):
        b = self.backend
        b.robot_state = 4
        self.assertFalse(b.alignment_control_ready())
        b.robot_state = 3
        self.assertTrue(b.alignment_control_ready())

    def test_arm_stream_status_changes_do_not_abort_alignment(self):
        from alignment import JointAlignment
        b = self.backend
        b.control_mode = 'arm_sdk'
        alignment = JointAlignment(b.names, self.q, self.q, 0.)
        for seq in range(40):
            b.robot_state = 3 if seq % 2 else 4
            alignment.step(self.q, np.zeros(29), seq, b.alignment_control_ready(), seq * .02)
        self.assertEqual(alignment.status, 'holding')


if __name__ == '__main__':
    unittest.main()
