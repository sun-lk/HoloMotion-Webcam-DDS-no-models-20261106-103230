import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retarget_config
from tracking_filter import OneEuro, PoseFilter, JointHold


def landmarks():
    p = np.zeros((33, 4)); p[:, 3] = .99
    for i, xyz in {11:[.2,-.5,0],12:[-.2,-.5,0],23:[.13,0,0],24:[-.13,0,0],
                   13:[.22,-.23,0],15:[.22,.02,-.01],14:[-.22,-.23,0],16:[-.22,.02,-.01]}.items():
        p[i, :3] = xyz
    return p


class TrackingFilterTests(unittest.TestCase):
    def setUp(self):
        self.cfg = retarget_config.load()['tracking']

    def test_static_noise_reduced_and_steps_remain_responsive(self):
        rng = np.random.default_rng(17)
        f = OneEuro(1., 3., 1.)
        raw = rng.normal(0., .003, 240)
        filtered = [float(f.update(x, i * .05)) for i, x in enumerate(raw)]
        self.assertLess(np.std(filtered[40:]), np.std(raw[40:]) * .55)
        reached = None
        for i in range(20):
            y = float(f.update(.2, 12. + i * .05))
            if y >= .18 and reached is None:
                reached = i * .05
        self.assertIsNotNone(reached)
        self.assertLessEqual(reached, .3)

    def test_timestamp_intervals_not_assumed_frame_rate(self):
        a, b = OneEuro(1., 0., 1.), OneEuro(1., 0., 1.)
        a.update(0., 0.); b.update(0., 0.)
        self.assertGreater(float(a.update(1., .1)), float(b.update(1., .02)))
        old = a.update(9., .1)
        self.assertLess(float(old), 1.)

    def test_isolated_spike_held_and_real_sustained_change_reacquired(self):
        f = PoseFilter(self.cfg, .5)
        p = landmarks()
        base = f.update(p, 0.)
        spike = p.copy(); spike[15, 0] += .8
        np.testing.assert_array_equal(f.update(spike, .05), base)
        self.assertTrue(f.held)
        f.update(p, .1)
        self.assertFalse(f.held)
        for i in range(3):
            f.update(spike, .15 + i * .05)
        self.assertFalse(f.held)

    def test_low_visibility_holds_briefly_but_cannot_refresh_validity(self):
        f = PoseFilter(self.cfg, .5)
        p = landmarks()
        base = f.update(p, 0.)
        p[15, 3] = 0.
        np.testing.assert_array_equal(f.update(p, .1), base)
        self.assertEqual(f.timestamp, 0.)
        with self.assertRaises(ValueError):
            f.update(p, .3)

    def test_global_translation_does_not_change_arm_directions(self):
        p = landmarks(); q = p.copy(); q[:, :3] += [1., 2., 3.]
        a, b = PoseFilter(self.cfg, .5), PoseFilter(self.cfg, .5)
        np.testing.assert_allclose(a.update(p, 0.), b.update(q, 0.), atol=1e-14)

    def test_gap_and_explicit_reset_remove_filter_history(self):
        f = PoseFilter(self.cfg, .5)
        p = landmarks(); f.update(p, 0.)
        p[15, 0] += .1
        filtered = f.update(p, 1.)
        np.testing.assert_allclose(filtered, p[:, :3] - (p[23, :3] + p[24, :3]) / 2.)
        f.reset()
        self.assertIsNone(f.timestamp)

    def test_deadband_holds_noise_and_releases_for_intentional_motion(self):
        h = JointHold(self.cfg)
        h.reset([0.])
        for i in range(20):
            h.update([.001 * (-1) ** i], i * .05)
        self.assertTrue(h.holding[0])
        held = h.accepted.copy()
        for i in range(20, 30):
            np.testing.assert_array_equal(h.update([.003 * (-1) ** i], i * .05), held)
        self.assertGreater(float(h.update([.1], 1.5)[0]), .09)
        self.assertFalse(h.holding[0])


if __name__ == '__main__':
    unittest.main()
