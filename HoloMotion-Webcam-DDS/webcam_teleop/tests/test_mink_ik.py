"""Real Mink/DAQP regressions; no policy, sockets or hardware."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import mink
import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'webcam_teleop'))
from retarget import WebcamRetarget
from arm_pose_fixtures import BASE, POSES, blend, errors


class MinkIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT/'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
        cls.robot = yaml.safe_load((ROOT/'checkpoints/HoloMotion_motion_tracking_model_v2.0.0/config.yaml').read_text())['robot']

    def adapter(self):
        r = self.robot
        a = WebcamRetarget(self.model, r['dof_names'], r['body_names'],
                          [r['init_state']['default_joint_angles'][n] for n in r['dof_names']])
        a.calibrate(BASE, strict=False)
        return a

    def test_gestures_from_rest_and_return_stay_bounded_and_continuous(self):
        for name, goal in POSES.items():
            with self.subTest(gesture=name):
                a = self.adapter()
                for i in range(300):
                    stamp = i*.02
                    p = blend(BASE, goal, float(np.clip((stamp-1)/2, 0, 1)))
                    self.checked_update(a, p, i, stamp)
                for arm_error in errors(a, goal).values():
                    self.assertLess(arm_error['upper_deg'], 2.)
                    self.assertLess(arm_error['fore_deg'], 2.)
                if name == 'hello_forearms_up':
                    self.assertGreater(a.ik.recovery_count, 0)
                    for _ in range(20):
                        self.checked_update(a, goal)
                    recovered = a.ik.recovery_count
                    for _ in range(20):
                        self.checked_update(a, goal)
                    self.assertEqual(a.ik.recovery_count, recovered)
                for i in range(300):
                    p = blend(goal, BASE, min(1., i*.02/2))
                    self.checked_update(a, p, 300+i, 6.+i*.02)
                for arm_error in errors(a, BASE).values():
                    self.assertLess(arm_error['upper_deg'], 3.)
                    self.assertLess(arm_error['fore_deg'], 3.)

    def checked_update(self, a, p, sample=None, stamp=None):
        before = a.q.copy()
        q, _, _ = a.update(p, dt=.02, sample_id=sample, timestamp=stamp)
        self.assertTrue(np.isfinite(q).all())
        self.assertLessEqual(float(np.max(np.abs(q-before))), .05000001)
        for arm in a.arms.values():
            angles = q[arm['indices']]
            self.assertTrue(np.all(angles >= arm['bounds'][:, 0]-.000001))
            self.assertTrue(np.all(angles <= arm['bounds'][:, 1]+.000001))

    def test_ik_freezes_root_legs_waist_wrists_and_never_edits_physics_model(self):
        ranges = self.model.jnt_range.copy()
        limited = self.model.jnt_limited.copy()
        a = self.adapter()
        # A measured, frozen wrist at its true limit must not make IK infeasible
        # merely because the eight moving joints have a margin.
        wrist = a.names.index('left_wrist_pitch_joint')
        a.q[wrist] = self.model.joint('left_wrist_pitch_joint').range[1]
        initial = a.q.copy()
        for _ in range(5):
            self.checked_update(a, POSES['hello_forearms_up'])
        moving = [i for arm in a.arms.values() for i in arm['indices']]
        fixed = [i for i in range(len(a.q)) if i not in moving]
        np.testing.assert_array_equal(a.q[fixed], initial[fixed])
        np.testing.assert_array_equal(a.ik.configuration.q[:7], a.d.qpos[:7])
        np.testing.assert_array_equal(self.model.jnt_range, ranges)
        np.testing.assert_array_equal(self.model.jnt_limited, limited)
        self.assertIsNot(a.ik.configuration.data, a.d)

    def test_failed_or_nonfinite_qp_never_changes_accepted_reference(self):
        for side_effect, returned in [(mink.exceptions.NoSolutionFound('test'), None),
                                      (None, np.full(self.model.nv, np.nan))]:
            a = self.adapter()
            before = a.q.copy()
            with patch('mink_ik.mink.solve_ik', side_effect=side_effect, return_value=returned):
                with self.assertRaises(ValueError):
                    a.update(BASE)
            np.testing.assert_array_equal(a.q, before)


if __name__ == '__main__':
    unittest.main()
