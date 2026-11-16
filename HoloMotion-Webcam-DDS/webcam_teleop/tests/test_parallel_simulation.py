"""Separate MuJoCo states, source selection and feedback tests with no network."""
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'webcam_teleop'))
from parallel_simulation import ParallelSimulation, choose_render_snapshot
from pose_http import HttpPoseSource


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
        d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, d, 0)
        self.ev = SimpleNamespace(m=self.m, d=d, target_dof_pos_by_name=dict(
            left_elbow_joint=.1, left_knee_joint=.2))
        self.calls = []
        def control(sleep=False):
            self.calls.append((self.ev.d, dict(self.ev.target_dof_pos_by_name)))
            mujoco.mj_step(self.m, self.ev.d)
        self.sim = ParallelSimulation(self.ev, control)

    def test_simulation_has_independent_physics_and_restores_evaluator(self):
        real_d = self.ev.d
        original_time = real_d.time
        self.sim.advance(['left_elbow_joint', 'left_knee_joint'], [.7, 1.1])
        self.assertIs(self.calls[0][0], self.sim.data)
        self.assertEqual(self.calls[0][1]['left_elbow_joint'], .7)
        self.assertEqual(self.calls[0][1]['left_knee_joint'], .2)
        self.assertIs(self.ev.d, real_d)
        self.assertEqual(real_d.time, original_time)
        self.assertGreater(self.sim.data.time, original_time)
        self.assertEqual(self.ev.target_dof_pos_by_name['left_elbow_joint'], .1)

    def test_feedback_select_and_reseed_from_actual_pose(self):
        real = self.ev.d
        real.qpos[2] = .91
        self.sim.seed(real)
        self.sim.select_feedback('simulation')
        self.assertIs(self.ev.d, self.sim.data)
        self.sim.data.qpos[2] = .8
        self.assertEqual(real.qpos[2], .91)
        self.sim.select_feedback('robot')
        self.assertIs(self.ev.d, real)

    def test_exception_restores_data_and_suppresses_invalid_preview(self):
        real = self.ev.d
        def bad(**kwargs):
            raise RuntimeError('injected simulation error')
        self.sim.control = bad
        with self.assertRaises(ValueError):
            self.sim.advance([], [])
        self.assertIs(self.ev.d, real)
        self.assertIsNone(self.sim.snapshot(np.zeros((30,3))))
        self.sim.seed(real)
        self.assertIsNotNone(self.sim.snapshot(np.zeros((30,3))))

    def test_render_selection_never_substitutes_a_simulated_robot(self):
        sim, real = object(), object()
        self.assertIs(choose_render_snapshot('simulation', sim, real), sim)
        self.assertIs(choose_render_snapshot('robot', sim, real), real)
        self.assertIsNone(choose_render_snapshot('robot', sim, None))

    def test_full_body_preview_uses_all_sent_targets(self):
        self.sim.advance(['left_elbow_joint', 'left_knee_joint'], [.7, 1.1], 'full_body')
        self.assertEqual(self.calls[0][1]['left_knee_joint'], 1.1)

    def test_http_render_pose_expires_and_repeated_seq_does_not_refresh(self):
        src = HttpPoseSource(self.m, 'http://unused')
        name = 'left_elbow_joint'
        payload = dict(names=[name], q=[.4], dq=[.1], imu_rpy=[0,0,0], seq=1, age_ms=1)
        with patch('pose_http.time.monotonic', return_value=10.):
            src._ingest(payload)
            self.assertIsNotNone(src.snapshot())
        with patch('pose_http.time.monotonic', return_value=11.):
            src._ingest(payload)
            self.assertIsNone(src.snapshot())
            self.assertFalse(src.status()['ok'])
        with self.assertRaises(ValueError):
            src._ingest(dict(payload, age_ms=1000))


if __name__ == '__main__':
    unittest.main()
