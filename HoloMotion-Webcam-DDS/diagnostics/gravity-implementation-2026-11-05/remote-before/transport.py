"""ZMQ 传输层（不依赖 ROS）。

ACTION 方向：远端 PUB bind  ->  本机 SUB connect（本机主动外连，天然穿 NAT）
STATE  方向：远端 SUB bind  <-  本机 PUB connect

两端都开 CONFLATE，只保留最新一帧：控制链路要的是"最新状态"而不是"完整历史"。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from . import protocol


@dataclass
class ReceivedAction:
    frame: protocol.ActionFrame
    recv_time: float  # time.monotonic()


class ActionReceiver:
    """后台线程收 ACTION 帧，只保留最新一帧。"""

    def __init__(
        self,
        endpoint: str,
        mode: str = "connect",
        topic: bytes = protocol.TOPIC_ACTION,
        recv_timeout_ms: int = 20,
    ) -> None:
        import zmq

        self._zmq = zmq
        self._topic = topic
        self._endpoint = endpoint
        self._mode = mode
        self._recv_timeout_ms = int(recv_timeout_ms)

        self._lock = threading.Lock()
        self._latest: Optional[ReceivedAction] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.received = 0
        self.decode_errors = 0
        self.last_error = ""

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SUBSCRIBE, self._topic)
        if self._mode == "bind":
            self._sock.bind(self._endpoint)
        else:
            self._sock.connect(self._endpoint)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="g1ra-action-recv", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._sock.recv()
            except self._zmq.Again:
                continue
            except Exception as exc:  # noqa: BLE001 - 传输异常不应拖垮主循环
                self.last_error = f"recv failed: {exc}"
                time.sleep(0.05)
                continue
            try:
                payload = protocol.strip_topic(raw, self._topic)
                frame = protocol.decode_action(payload)
            except protocol.ProtocolError as exc:
                self.decode_errors += 1
                self.last_error = str(exc)
                continue
            with self._lock:
                self._latest = ReceivedAction(frame=frame, recv_time=time.monotonic())
            self.received += 1

    def latest(self) -> Optional[ReceivedAction]:
        with self._lock:
            return self._latest

    def age(self) -> float:
        with self._lock:
            if self._latest is None:
                return float("inf")
            return time.monotonic() - self._latest.recv_time

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._sock.close(0)
        except Exception:  # noqa: BLE001
            pass


class StatePublisher:
    """把 STATE 帧发给远端。"""

    def __init__(
        self,
        endpoint: str,
        mode: str = "connect",
        topic: bytes = protocol.TOPIC_STATE,
        send_hwm: int = 2,
    ) -> None:
        import zmq

        self._topic = topic
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, int(send_hwm))
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        if mode == "bind":
            self._sock.bind(endpoint)
        else:
            self._sock.connect(endpoint)
        self.sent = 0
        self.last_error = ""

    def send(self, frame: protocol.StateFrame) -> None:
        payload = protocol.encode_state(
            seq=frame.seq,
            stamp_ns=frame.stamp_ns,
            q=frame.q,
            dq=frame.dq,
            quat_wxyz=frame.quat_wxyz,
            gyro=frame.gyro,
            robot_state=frame.robot_state,
            bridge_flags=frame.bridge_flags,
            wireless=getattr(frame, "wireless", b""),
        )
        try:
            self._sock.send(self._topic + payload, flags=0, copy=True)
            self.sent += 1
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)

    def close(self) -> None:
        try:
            self._sock.close(0)
        except Exception:  # noqa: BLE001
            pass


def make_state_frame(
    seq: int,
    stamp_ns: int,
    q,
    dq,
    quat_wxyz,
    gyro,
    robot_state: int,
    bridge_flags: int,
    wireless: bytes = b"",
) -> protocol.StateFrame:
    import numpy as np

    return protocol.StateFrame(
        seq=int(seq),
        stamp_ns=int(stamp_ns),
        q=np.asarray(q, dtype=np.float32),
        dq=np.asarray(dq, dtype=np.float32),
        quat_wxyz=np.asarray(quat_wxyz, dtype=np.float32),
        gyro=np.asarray(gyro, dtype=np.float32),
        robot_state=int(robot_state),
        bridge_flags=int(bridge_flags),
        wireless=bytes(wireless),
    )


def now_ns() -> int:
    return time.monotonic_ns()


def _unused(_: Tuple = ()) -> None:  # pragma: no cover
    return None
