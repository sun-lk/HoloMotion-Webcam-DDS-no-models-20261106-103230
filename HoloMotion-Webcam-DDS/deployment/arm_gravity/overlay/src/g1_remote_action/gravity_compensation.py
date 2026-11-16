"""Static gravity feedforward. No SDK, transport, or robot writes in this module.

Measured body/finger positions and pelvis IMU orientation feed fixed-base
Pinocchio models. Alignment uses full gravity; learned policy uses only the
Dex3-minus-rubber difference. This is not inverse dynamics or payload fitting.
"""
from pathlib import Path

import numpy as np

from . import protocol

FINGERS = [f'{side}_hand_{finger}_{i}_joint'
           for side in ('left', 'right')
           for finger, count in (('thumb', 3), ('index', 2), ('middle', 2))
           for i in range(count)]

DEFAULTS = dict(enabled=False, full_model='../models/gravity/g1_dex3.urdf',
                nominal_model='../models/gravity/g1_rubber.urdf',
                scale=1.0, max_torque_fraction=0.3, torque_rate_nm_s=3.0,
                compute_hz=50.0, body_timeout_s=0.1, hand_timeout_s=0.25)


def settings(cfg, config_path):
    result = dict(DEFAULTS)
    raw = cfg.get('gravity') or {}
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown gravity settings: {sorted(unknown)}')
    result.update(raw)
    if not isinstance(result['enabled'], bool):
        raise ValueError('gravity.enabled must be boolean')
    for key in ('scale', 'max_torque_fraction', 'torque_rate_nm_s',
                'compute_hz', 'body_timeout_s', 'hand_timeout_s'):
        value = float(result[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'gravity.{key} must be finite and positive')
        result[key] = value
    if result['scale'] > 1 or result['max_torque_fraction'] > 1:
        raise ValueError('gravity scale/fraction must be <= 1')
    for key in ('full_model', 'nominal_model'):
        result[key] = str((Path(config_path).resolve().parent / result[key]).resolve())
    return result


def vector(values, n, name):
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n,) or not np.isfinite(arr).all():
        raise ValueError(f'invalid {name}: expected finite ({n},)')
    return arr


def body_gravity(quat):
    w, x, y, z = vector(quat, 4, 'quaternion')
    norm = np.linalg.norm([w, x, y, z])
    if not 0.5 < norm < 1.5:
        raise ValueError('invalid quaternion norm')
    w, x, y, z = np.array([w, x, y, z]) / norm
    # Third row of R_world_pelvis, multiplied by world gravity.
    return -9.81 * np.array([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])


class GravityModels:
    def __init__(self, cfg, names):
        import pinocchio as pin

        self.pin = pin
        self.names = list(names)
        if len(self.names) != 29 or len(set(self.names)) != 29:
            raise ValueError('expected 29 unique body joint names')
        self.models = []
        for key, expected in (('full_model', self.names + FINGERS),
                              ('nominal_model', self.names)):
            model = pin.buildModelFromUrdf(cfg[key])
            actual = set(model.names[1:])
            if actual != set(expected):
                raise ValueError(f'{key} joint mismatch: {actual.symmetric_difference(expected)}')
            joints = []
            for name in expected:
                joint = model.joints[model.getJointId(name)]
                if joint.nq != 1 or joint.nv != 1:
                    raise ValueError(f'{name} must be a scalar revolute joint')
                joints.append((name, joint.idx_q, joint.idx_v))
            self.models.append((model, model.createData(), joints))

    def compute(self, body, fingers, quat):
        body = vector(body, 29, 'body q')
        fingers = vector(fingers, 14, 'finger q')
        values = dict(zip(self.names + FINGERS, np.r_[body, fingers]))
        gravity = body_gravity(quat)
        results = []
        for model, data, joints in self.models:
            q = np.zeros(model.nq)
            for name, iq, _ in joints:
                q[iq] = values[name]
            model.gravity.linear = gravity
            tau = self.pin.computeGeneralizedGravity(model, data, q)
            by_name = {name: float(tau[iv]) for name, _, iv in joints}
            results.append(vector([by_name[n] for n in self.names[15:29]], 14, 'gravity torque'))
        return results[0], results[0] - results[1]


class GravityCompensator:
    def __init__(self, cfg, spec, models=None):
        self.cfg = cfg
        self.enabled = cfg['enabled']
        self.models = models
        self.tau = np.zeros(14)
        self.target = np.zeros(14)
        self.phase = protocol.PHASE_UNSPECIFIED
        self.reason = 'disabled'
        self.clipped = False
        self._last_compute = float('-inf')
        self._last_phase = None
        self.caps = np.zeros(14)
        if self.enabled:
            if list(spec.motor_index) != list(range(29)):
                raise ValueError('arm bridge gravity requires body order = motor order 0..28')
            effort = vector(spec.effort, 29, 'effort limits')
            if np.any(effort <= 0):
                raise ValueError('effort limits must be positive')
            self.caps = effort[15:29] * cfg['max_torque_fraction']
            if self.models is None:
                self.models = GravityModels(cfg, spec.names)

    def update(self, phase, active, body, fingers, quat, body_age, hand_age, now, dt):
        desired_phase = self.phase if phase == protocol.PHASE_HOLD else phase
        valid = (self.enabled and active and desired_phase in
                 (protocol.PHASE_ALIGNMENT, protocol.PHASE_POLICY))
        if valid:
            try:
                if not (0 <= body_age <= self.cfg['body_timeout_s'] and
                        0 <= hand_age <= self.cfg['hand_timeout_s']):
                    raise ValueError('stale measured body/hand state')
                # Validate every control tick, including ticks between model evaluations.
                vector(body, 29, 'body q')
                vector(fingers, 14, 'finger q')
                body_gravity(quat)
                if (desired_phase != self._last_phase or
                        now - self._last_compute >= 1/self.cfg['compute_hz']):
                    full, delta = self.models.compute(body, fingers, quat)
                    raw = vector(full if desired_phase == protocol.PHASE_ALIGNMENT else delta,
                                 14, 'feedforward') * self.cfg['scale']
                    self.target = np.clip(raw, -self.caps, self.caps)
                    self.clipped = bool(np.any(np.abs(raw) > self.caps))
                    self._last_compute = now
                    self._last_phase = desired_phase
                self.phase = desired_phase
                self.reason = 'full' if desired_phase == protocol.PHASE_ALIGNMENT else 'delta'
            except Exception as exc:
                valid = False
                self.reason = f'invalid state/model: {exc}'
        else:
            self.reason = 'disabled' if not self.enabled else 'inactive/unspecified'
        if not valid:
            self.target.fill(0)
            self.phase = protocol.PHASE_UNSPECIFIED
            self._last_phase = None
            self.clipped = False
        # Cap elapsed time so a delayed iteration never produces a large jump.
        step = self.cfg['torque_rate_nm_s'] * np.clip(dt, 0, 0.02)
        self.tau += np.clip(self.target-self.tau, -step, step)
        return self.tau.copy()

    def status(self):
        return dict(enabled=self.enabled, phase=self.phase, reason=self.reason,
                    clipped=self.clipped, tau_nm=np.round(self.tau, 4).tolist(),
                    target_nm=np.round(self.target, 4).tolist())
