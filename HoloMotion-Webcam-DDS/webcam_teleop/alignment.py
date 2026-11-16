"""Frozen human target -> bounded shoulder/elbow alignment, without policy inference."""
import numpy as np


class JointAlignment:
    SPEED = .25  # rad/s; reference loop is 50 Hz.
    ANGLE_TOL = .05  # rad, about 2.9 degrees.
    VELOCITY_TOL = .08  # rad/s.
    SETTLE_TIME = .6

    def __init__(self, names, actual, human_target, now):
        self.indices = [i for i, n in enumerate(names) if 'shoulder_' in n or 'elbow_' in n]
        if len(self.indices) != 8:
            raise ValueError('对齐仅支持双臂共 8 个肩肘关节')
        actual = np.asarray(actual, dtype=float)
        target = np.asarray(human_target, dtype=float)
        if actual.shape != (len(names),) or target.shape != actual.shape or not np.isfinite(np.r_[actual, target]).all():
            raise ValueError('对齐目标或实测角无效')
        self.command = actual.copy()
        self.target = actual.copy()
        self.target[self.indices] = target[self.indices]
        self.started = self.last_tick = now
        self.settle_since = None
        self.last_seq = None
        self.status = 'waiting_policy'
        self.error = self.speed = None

    def step(self, actual, velocity, seq, policy_active, now):
        actual, velocity = np.asarray(actual), np.asarray(velocity)
        if actual.shape != self.target.shape or velocity.shape != self.target.shape or not np.isfinite(np.r_[actual, velocity]).all():
            raise ValueError('对齐反馈无效，已停止')
        dt = min(.02, max(0., now - self.last_tick))
        self.last_tick = now
        self.error = float(np.max(np.abs(self.target[self.indices] - actual[self.indices])))
        self.speed = float(np.max(np.abs(velocity[self.indices])))
        if not policy_active:
            self.settle_since = None
            if self.status != 'waiting_policy':
                raise ValueError('全身低层桥已退出 POLICY，对齐已停止；请重新对齐')
        else:
            self.command[self.indices] += np.clip(
                self.target[self.indices] - self.command[self.indices], -self.SPEED * dt, self.SPEED * dt)
            if self.status != 'holding' or self.error > self.ANGLE_TOL or self.speed > self.VELOCITY_TOL:
                self.status = 'moving'
            # Repeated frames cannot establish arrival; require advancing feedback.
            if seq != self.last_seq:
                command_ready = np.max(np.abs(self.target[self.indices] - self.command[self.indices])) <= self.ANGLE_TOL
                if command_ready and self.error <= self.ANGLE_TOL and self.speed <= self.VELOCITY_TOL:
                    if self.settle_since is None:
                        self.settle_since = now
                    elif now - self.settle_since >= self.SETTLE_TIME:
                        self.status = 'holding'
                else:
                    self.settle_since = None
        self.last_seq = seq
        return self.command.copy()
