"""Exercise SDK command mapping with fake messages/publishers; no DDS imports."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from g1_remote_action.arm_backend import ArmLink


class ArmGravityTests(unittest.TestCase):
    def make_link(self):
        link = ArmLink.__new__(ArmLink)
        link._msg = NS(motor_cmd=[NS(mode=0, q=0., dq=0., tau=0., kp=0., kd=0.) for _ in range(35)])
        link._latest = NS(mode_machine=5)
        link._crc = NS(Crc=lambda msg: 123)
        self.writes = []
        link._pub = NS(Write=lambda msg: self.writes.append(msg))
        link._kp_high, link._kd_high = 30., .91
        link._kp_wrist, link._kd_wrist = 30., .91
        link._waist_kp, link._waist_kd = 120., 2.
        link.tx_count = 0
        link.rx_count = 0
        link.last_error = ''
        link._hand_samples = [(None, float('-inf')), (None, float('-inf'))]
        return link

    def test_tau_maps_only_to_arms_preserving_gains_waist_and_legs(self):
        link = self.make_link()
        tau = np.linspace(-1, 1, 14)
        link.publish(np.arange(14)/10, [.1, .2, .3], .7, tau)
        self.assertEqual(len(self.writes), 1)
        for j in range(12):
            self.assertEqual(link._msg.motor_cmd[j].mode, 0)
        for j in range(12, 15):
            c = link._msg.motor_cmd[j]
            self.assertEqual((c.tau, c.kp, c.kd), (0., 120., 2.))
        for i, j in enumerate(range(15, 29)):
            c = link._msg.motor_cmd[j]
            self.assertEqual((c.tau, c.kp, c.kd), (tau[i], 30., .91))
        self.assertEqual(link._msg.motor_cmd[29].q, .7)
        link.publish(np.zeros(14), np.zeros(3), 1.)
        self.assertTrue(all(link._msg.motor_cmd[j].tau == 0 for j in range(15,29)))

    def test_measured_sample_mapping_and_local_freshness(self):
        link = self.make_link()
        body = NS(motor_state=[NS(q=i/10) for i in range(35)],
                  imu_state=NS(quaternion=[1.,0.,0.,0.]))
        with patch('g1_remote_action.arm_backend.time.monotonic', return_value=10.):
            link._on_state(body)
            link._on_hand(NS(motor_state=[NS(q=i) for i in range(7)]), 0)
            link._on_hand(NS(motor_state=[NS(q=-i) for i in range(7)]), 1)
        q, fingers, quat, body_age, hand_age = link.gravity_sample(10.02)
        np.testing.assert_array_equal(q, np.arange(29)/10)
        np.testing.assert_array_equal(fingers, np.r_[np.arange(7), -np.arange(7)])
        self.assertAlmostEqual(body_age, .02)
        self.assertAlmostEqual(hand_age, .02)
        link._on_hand(NS(motor_state=[NS(q=np.nan)]*7), 0)
        self.assertIsNone(link.gravity_sample(10.03)[1])


if __name__ == '__main__':
    unittest.main()
