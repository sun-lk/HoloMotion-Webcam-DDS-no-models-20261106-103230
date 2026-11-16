"""Timestamped pose smoothing and reference deadband; no motor control."""
import numpy as np


class OneEuro:
    def __init__(self, min_cutoff, beta, derivative_cutoff):
        self.min_cutoff, self.beta, self.derivative_cutoff = min_cutoff, beta, derivative_cutoff
        self.value = self.raw = self.velocity = self.time = None

    @staticmethod
    def alpha(cutoff, dt):
        return 1. / (1. + 1. / (2. * np.pi * cutoff * dt))

    def update(self, value, timestamp):
        value = np.asarray(value, dtype=float)
        if self.time is None:
            self.value = value.copy()
            self.raw = value.copy()
            self.velocity = np.zeros_like(value)
        else:
            dt = timestamp - self.time
            if dt <= 0:
                return self.value.copy()
            a = self.alpha(self.derivative_cutoff, dt)
            self.velocity += a * ((value - self.raw) / dt - self.velocity)
            a = self.alpha(self.min_cutoff + self.beta * np.abs(self.velocity), dt)
            self.value += a * (value - self.value)
            self.raw = value.copy()
        self.time = timestamp
        return self.value.copy()


class PoseFilter:
    REQUIRED = [11, 12, 13, 14, 15, 16, 23, 24]

    def __init__(self, cfg, min_visibility):
        self.cfg, self.min_visibility = cfg, min_visibility
        self.reset()

    def reset(self):
        c = self.cfg
        self.filter = OneEuro(c['min_cutoff_hz'], c['beta'], c['derivative_cutoff_hz'])
        self.accepted = self.filtered = self.timestamp = None
        self.candidate = None
        self.candidate_count = 0
        self.held = False
        self.rejected = 0

    def update(self, landmarks, timestamp):
        p = np.asarray(landmarks, dtype=float)
        if p.shape != (33, 4) or not np.isfinite(p).all() or not np.isfinite(timestamp):
            raise ValueError('人体关键点或采集时间无效')
        if self.timestamp is not None and timestamp <= self.timestamp:
            self.held = True
            return self.filtered.copy()
        # Translation is irrelevant to arm directions; remove noisy global hip motion.
        xyz = p[:, :3] - (p[23, :3] + p[24, :3]) / 2.
        visible = np.min(p[self.REQUIRED, 3]) >= self.min_visibility
        gap = None if self.timestamp is None else timestamp - self.timestamp
        outlier = False
        if visible and gap is not None and gap <= self.cfg['reset_gap_s']:
            jump = np.max(np.linalg.norm(xyz[self.REQUIRED] - self.accepted[self.REQUIRED], axis=1))
            outlier = jump > self.cfg['jump_margin_m'] + self.cfg['max_landmark_speed_m_s'] * gap
        if visible and outlier:
            # A single spike is held; three consistent new samples may be real movement.
            consistent = self.candidate is not None and np.max(np.linalg.norm(
                xyz[self.REQUIRED] - self.candidate[self.REQUIRED], axis=1)) < .05
            self.candidate_count = self.candidate_count + 1 if consistent else 1
            self.candidate = xyz.copy()
            outlier = self.candidate_count < 3
        if not visible or outlier:
            self.rejected += 1
            self.held = True
            if not visible:
                self.candidate = None
                self.candidate_count = 0
            if self.filtered is not None and gap <= self.cfg['confidence_hold_s']:
                return self.filtered.copy()
            raise ValueError('人体关键点持续不可靠，请恢复肩肘腕髋可见后重新开始')
        if gap is not None and gap > self.cfg['reset_gap_s']:
            self.filter = OneEuro(self.cfg['min_cutoff_hz'], self.cfg['beta'], self.cfg['derivative_cutoff_hz'])
        self.accepted = xyz.copy()
        self.timestamp = timestamp
        self.filtered = self.filter.update(xyz, timestamp)
        self.candidate = None
        self.candidate_count = 0
        self.held = False
        return self.filtered.copy()


class JointHold:
    def __init__(self, cfg):
        self.cfg = cfg
        self.accepted = None

    def reset(self, value):
        self.accepted = np.asarray(value, dtype=float).copy()
        self.origin = self.accepted.copy()
        self.since = np.full(self.accepted.shape, np.nan)
        self.holding = np.zeros(self.accepted.shape, dtype=bool)

    def update(self, value, timestamp):
        value = np.asarray(value, dtype=float)
        if self.accepted is None:
            self.reset(value)
        release = self.holding & (np.abs(value - self.accepted) > self.cfg['hold_exit_rad'])
        self.holding[release] = False
        moving = ~self.holding
        restart = moving & (np.isnan(self.since) | (np.abs(value - self.origin) > self.cfg['hold_enter_rad']))
        self.origin[restart] = value[restart]
        self.since[restart] = timestamp
        self.accepted[moving] = value[moving]
        lock = moving & ((timestamp - self.since) >= self.cfg['hold_settle_s'])
        self.holding[lock] = True
        return self.accepted.copy()
