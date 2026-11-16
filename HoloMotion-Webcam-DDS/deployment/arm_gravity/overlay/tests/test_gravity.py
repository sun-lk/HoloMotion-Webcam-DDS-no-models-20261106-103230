"""Non-actuating tests: fake dynamics plus optional real Pinocchio model tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from g1_remote_action import protocol as p
from g1_remote_action.config import load_config
from g1_remote_action.limits import build_spec
from g1_remote_action.core import ActionProcessor
from g1_remote_action.transport import ReceivedAction
from g1_remote_action.gravity_compensation import GravityCompensator, GravityModels, settings, body_gravity


class GravityTests(unittest.TestCase):
    def setUp(self):
        self.spec = build_spec(load_config(ROOT / 'config/bridge.yaml'))
        self.cfg = settings({'gravity': {'enabled': True}}, ROOT / 'config/bridge.yaml')
        self.fake = SimpleNamespace(compute=lambda *args: (np.ones(14)*4, np.ones(14)*.5))
        self.g = GravityCompensator(self.cfg, self.spec, self.fake)

    def tick(self, phase=1, active=True, **kw):
        args = dict(phase=phase, active=active, body=np.zeros(29), fingers=np.zeros(14),
                    quat=[1, 0, 0, 0], body_age=0., hand_age=0., now=0., dt=.004)
        args.update(kw)
        return self.g.update(**args)

    def test_slew_limit_and_long_pause(self):
        np.testing.assert_allclose(self.tick(), .012)
        np.testing.assert_allclose(self.tick(dt=10), .072)

    def test_full_delta_and_caps(self):
        self.tick()
        np.testing.assert_allclose(self.g.target, np.minimum(4, self.g.caps))
        self.assertTrue(self.g.clipped)  # 5 Nm wrists -> 1.5 Nm caps
        self.tick(2)
        np.testing.assert_allclose(self.g.target, .5)
        self.assertFalse(self.g.clipped)

    def test_transition_has_no_torque_jump(self):
        for i in range(100):
            self.tick(now=i*.004)
        old = self.g.tau.copy()
        new = self.tick(2, now=.4)
        self.assertLessEqual(np.max(np.abs(new-old)), .012000001)

    def test_hold_remembers_phase_and_fault_clears_it(self):
        self.tick(2)
        self.tick(3)
        self.assertEqual(self.g.phase, 2)
        self.tick(3, hand_age=1)
        self.assertEqual(self.g.phase, 0)
        self.tick(3)
        np.testing.assert_array_equal(self.g.target, 0)

    def test_inactive_legacy_and_invalid_inputs_fade_to_zero(self):
        for kwargs in ({'active': False}, {'phase': 0}, {'body_age': 1},
                       {'hand_age': float('nan')}, {'fingers': None},
                       {'body': np.full(29, np.nan)}, {'quat': [0, 0, 0, 0]}):
            self.tick()
            old = self.g.tau.copy()
            self.tick(**kwargs)
            np.testing.assert_array_equal(self.g.target, 0)
            self.assertTrue(np.all(np.abs(self.g.tau) <= np.abs(old)))

    def test_disabled_does_not_load_pinocchio(self):
        cfg = settings({}, ROOT / 'config/bridge.yaml')
        self.g = GravityCompensator(cfg, self.spec)
        np.testing.assert_array_equal(self.tick(), 0)

    def test_model_exception_fades_and_recovers(self):
        def bad(*args):
            raise RuntimeError('injected failure')
        self.tick()
        self.fake.compute = bad
        self.tick(now=1)
        self.assertIn('injected failure', self.g.reason)
        np.testing.assert_array_equal(self.g.target, 0)

    def test_quaternion_body_frame_and_sign(self):
        np.testing.assert_allclose(body_gravity([1,0,0,0]), [0,0,-9.81])
        np.testing.assert_allclose(body_gravity([np.sqrt(.5),0,np.sqrt(.5),0]), [9.81,0,0], atol=1e-12)
        np.testing.assert_allclose(body_gravity([-1,0,0,0]), [0,0,-9.81])

    def test_config_rejects_bad_values(self):
        for values in ({'enabled': 'false'}, {'scale': 2}, {'torque_rate_nm_s': 0},
                       {'compute_hz': float('nan')}, {'oops': 1}):
            with self.assertRaises(ValueError):
                settings({'gravity': values}, ROOT / 'config/bridge.yaml')

    def test_protocol_compatible_and_phase_not_changed_by_duplicate(self):
        proc = ActionProcessor(self.spec, min_valid_frames=1)
        def rec(seq, phase):
            raw = p.encode_action(seq, 0, self.spec.default, self.spec.kp, self.spec.kd, phase=phase)
            self.assertEqual(len(raw), 380)
            return ReceivedAction(p.decode_action(raw), 1.)
        self.assertEqual(proc.process(rec(1, 1), 1., .004).phase, 1)
        self.assertEqual(proc.process(rec(1, 2), 1., .004).phase, 1)
        self.assertEqual(proc.process(rec(2, 2), 1., .004).phase, 2)
        self.assertEqual(proc.process(rec(1, 1), 1., .004).reason, 'seq_rollback')
        self.assertEqual(proc.process(rec(2, 1), 1., .004).phase, 2)
        self.assertEqual(p.decode_action(p.encode_action(3,0,self.spec.default,self.spec.kp,self.spec.kd)).phase, 0)
        with self.assertRaises(p.ProtocolError):
            rec(4, 5)


class PinocchioTests(unittest.TestCase):
    def test_saved_pose_matches_independent_urdf_gravity(self):
        try:
            import pinocchio
        except ImportError:
            self.skipTest('target-runtime Pinocchio required')
        state = json.loads((ROOT / 'tests/gravity_fixture/state.json').read_text())
        audit = json.loads((ROOT / 'tests/gravity_fixture/audit.json').read_text())
        spec = build_spec(load_config(ROOT / 'config/bridge.yaml'))
        cfg = settings({}, ROOT / 'config/bridge.yaml')
        models = GravityModels(cfg, spec.names)
        r, pitch, y = np.asarray(state['imu_rpy'])/2
        cr, cp, cy = np.cos([r, pitch, y])
        sr, sp, sy = np.sin([r, pitch, y])
        quat = [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
        full, delta = models.compute(state['q'], state['hand_q'], quat)
        ref = audit['arm_gravity_at_saved_pose_nm']
        np.testing.assert_allclose(full, [ref[n]['dex3'] for n in spec.names[15:29]], atol=1e-9)
        np.testing.assert_allclose(delta, [ref[n]['difference'] for n in spec.names[15:29]], atol=1e-9)
        # Independent finite difference of gravitational potential at a tilted,
        # asymmetric pose catches torque sign, model order, and IMU rotation.
        from g1_remote_action.gravity_compensation import FINGERS
        values = dict(zip(spec.names + FINGERS, state['q'] + state['hand_q']))
        for model, data, joints in models.models:
            q = np.zeros(model.nq)
            for name, iq, iv in joints:
                q[iq] = values[name]
            expected = models.pin.computeGeneralizedGravity(model, data, q).copy()
            for name, iq, iv in joints:
                if name not in spec.names[15:29]:
                    continue
                qp, qm = q.copy(), q.copy()
                qp[iq] += 1e-6
                qm[iq] -= 1e-6
                numerical = (models.pin.computePotentialEnergy(model,data,qp) -
                             models.pin.computePotentialEnergy(model,data,qm))/2e-6
                self.assertAlmostEqual(numerical, expected[iv], places=6)


if __name__ == '__main__':
    unittest.main()
