"""Synthetic feedback only; no socket, policy, or robot commands."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment import JointAlignment


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.names = ['waist_yaw_joint'] + [f'{side}_{joint}_joint'
            for side in ('left', 'right')
            for joint in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll')]
        self.actual = np.linspace(-.1, .1, len(self.names))
        self.human = self.actual + .2
        self.alignment = JointAlignment(self.names, self.actual, self.human, 0.)

    def test_only_shoulders_and_elbows_move_and_delay_does_not_enlarge_step(self):
        a = self.alignment
        q = a.step(self.actual, np.zeros(len(self.names)), 1, True, 1.)
        self.assertLessEqual(float(np.max(np.abs(q - self.actual))), .00500001)
        fixed = [i for i in range(len(self.names)) if i not in a.indices]
        np.testing.assert_array_equal(q[fixed], self.actual[fixed])

    def test_waiting_for_policy_only_sends_starting_hold(self):
        a = self.alignment
        q = a.step(self.actual, np.zeros(len(self.names)), 1, False, .1)
        np.testing.assert_array_equal(q, self.actual)
        self.assertEqual(a.status, 'waiting_policy')

    def test_arrival_requires_low_speed_and_sustained_advancing_feedback(self):
        a = JointAlignment(self.names, self.actual, self.actual, 0.)
        zeros = np.zeros(len(self.names))
        for seq in range(5):
            a.step(self.actual, np.ones(len(self.names)), seq, True, seq * .02)
        self.assertNotEqual(a.status, 'holding')
        for seq in range(5, 45):
            a.step(self.actual, zeros, seq, True, seq * .02)
        self.assertEqual(a.status, 'holding')
        a.step(self.actual, zeros, 44, True, .9)
        self.assertEqual(a.status, 'holding')

    def test_repeated_state_cannot_confirm_arrival(self):
        a = JointAlignment(self.names, self.actual, self.actual, 0.)
        for i in range(50):
            a.step(self.actual, np.zeros(len(self.names)), 1, True, i * .02)
        self.assertNotEqual(a.status, 'holding')

    def test_large_error_or_unfinished_command_prevents_arrival(self):
        a = self.alignment
        for i in range(40):
            a.step(self.actual, np.zeros(len(self.names)), i, True, i * .02)
        self.assertNotEqual(a.status, 'holding')
        a = JointAlignment(self.names, self.actual, self.actual + 2., 0.)
        for i in range(40):
            a.step(a.target, np.zeros(len(self.names)), i, True, i * .02)
        self.assertNotEqual(a.status, 'holding')

    def test_policy_exit_stops_but_elapsed_time_does_not(self):
        a = self.alignment
        a.step(self.actual, np.zeros(len(self.names)), 1, True, .02)
        with self.assertRaises(ValueError):
            a.step(self.actual, np.zeros(len(self.names)), 2, False, .04)
        a = JointAlignment(self.names, self.actual, self.human, 0.)
        q = a.step(self.actual, np.zeros(len(self.names)), 1, True, 100.)
        self.assertEqual(a.status, 'moving')
        self.assertGreater(a.error, a.ANGLE_TOL)
        self.assertLessEqual(float(np.max(np.abs(q - self.actual))), .00500001)

    def test_new_alignment_is_absolute_not_accumulated(self):
        actual = self.actual + .1
        a = JointAlignment(self.names, actual, self.human, 0.)
        np.testing.assert_array_equal(a.target[a.indices], self.human[a.indices])
        np.testing.assert_array_equal(a.command, actual)
        self.human[:] = 99.
        self.assertLess(float(np.max(a.target)), 1.)

    def test_nonfinite_feedback_rejected(self):
        actual = self.actual.copy()
        actual[1] = np.nan
        with self.assertRaises(ValueError):
            self.alignment.step(actual, np.zeros(len(self.names)), 1, True, .02)


if __name__ == '__main__':
    unittest.main()
