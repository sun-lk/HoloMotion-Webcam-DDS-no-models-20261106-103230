"""Bounded asynchronous joint telemetry; records observations without gating motion."""
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import queue
import threading
import time


class MotionDiagnostics:
    def __init__(self, path):
        self.path = str(Path(path))
        self._queue = queue.Queue(maxsize=128)
        self._last = -float('inf')
        self.dropped = 0
        self._thread = threading.Thread(target=self._write, daemon=True)
        self._thread.start()

    def record(self, transport, phase, reference=None, alignment=None, ik=None):
        now = time.monotonic()
        if now - self._last < 1. or transport._state is None:
            return
        self._last = now
        frame = transport._state
        record = dict(time_unix_s=time.time(), monotonic_s=now, phase=phase,
            state_seq=transport.state_seq, state_age_ms=max(0., (now - transport._state_time) * 1000.),
            control_mode=transport.control_mode, robot_state=transport.robot_state,
            joint_names=list(transport.names), actual_q_rad=list(map(float, frame.q)),
            actual_dq_rad_s=list(map(float, frame.dq)),
            command_q_rad=None if transport._last_command is None else list(map(float, transport._last_command)),
            reference_q_rad=None if reference is None else list(map(float, reference)),
            frozen_target_q_rad=None if alignment is None else list(map(float, alignment.target)),
            max_alignment_error_rad=None if alignment is None else alignment.error,
            max_alignment_speed_rad_s=None if alignment is None else alignment.speed,
            observed_settled=None if alignment is None else alignment.status == 'holding',
            bridge_flags=int(getattr(frame, 'bridge_flags', 0)),
            feedback_source=getattr(transport, 'feedback_source', 'robot'),
            simulated_q_rad=(None if (getattr(transport, 'simulation', None) is None or
                                     transport.simulation.error) else
                             list(map(float, transport.simulation.data.qpos[transport.qpos_idx]))),
            ik=ik,
            dropped_records=self.dropped)
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            handler = RotatingFileHandler(self.path, maxBytes=25 * 1024 * 1024,
                                          backupCount=2, encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(message)s'))
        except OSError as exc:
            print(f'[diagnostics] 无法打开日志：{exc}', flush=True)
            return
        try:
            while True:
                record = self._queue.get()
                if record is None:
                    break
                line = json.dumps(record, ensure_ascii=False, allow_nan=False)
                handler.handle(logging.LogRecord('motion', logging.INFO, '', 0, line, (), None))
        finally:
            handler.close()

    def close(self):
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            return
        self._thread.join(timeout=2.)
