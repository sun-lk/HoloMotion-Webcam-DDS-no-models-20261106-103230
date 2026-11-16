"""Fail-closed preflight for policy targets; never opens network sockets."""
import time
import numpy as np


class SimulationGate:
    def __init__(self, settle_s=1.0, max_speed=8.0, max_error=0.35):
        self.settle_s = settle_s
        self.max_speed = max_speed
        self.max_error = max_error
        self.reset()

    def reset(self):
        self.since = None
        self.last = None
        self.reason = '等待连续仿真验证'

    def check(self, q, dq, target, limits, height, now=None):
        now = time.monotonic() if now is None else now
        q, dq, target, limits = map(np.asarray, (q, dq, target, limits))
        valid = (q.ndim == 1 and dq.shape == q.shape == target.shape
                 and limits.shape == (q.size, 2) and q.size > 0)
        if valid:
            valid = (np.isfinite(q).all() and np.isfinite(dq).all()
                     and np.isfinite(target).all() and np.isfinite(limits).all()
                     and np.isfinite(height) and height >= .45
                     and np.all(limits[:, 0] < limits[:, 1])
                     and np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1])
                     and np.all(target >= limits[:, 0]) and np.all(target <= limits[:, 1])
                     and np.max(np.abs(dq)) <= self.max_speed
                     and np.max(np.abs(q-target)) <= self.max_error)
        if not valid:
            self.reset()
            self.reason = '仿真未通过：检查跌倒、关节越界、速度及跟踪误差'
            return False
        if self.last is None or now <= self.last or now-self.last > .1:
            self.since = now
        self.last = now
        ready = now-self.since >= self.settle_s
        self.reason = '仿真验证通过' if ready else '等待连续1秒有效仿真（间隔不超过100ms）'
        return ready
