#!/usr/bin/env python3
"""把 HoloMotion 的 /humanoid/action 转成 g1-remote-action 的 G1RA ACTION 帧。

机器人侧由 g1-remote-action 的桥（ROS 或 SDK 后端）负责校验、限幅、
限速并驱动 main_node / rt/lowcmd；本节点只做线协议转换和状态回灌。

订阅（机器人本体经 g1_remote_action.lowstate_relay 转发的 ROS 话题）：
  - <action_topic>   std_msgs/Float32MultiArray，29 维绝对目标关节角
  - <kps_topic>      std_msgs/Float32MultiArray，29 维刚度
  - <kds_topic>      std_msgs/Float32MultiArray，29 维阻尼

发布：
  - ZMQ <action_endpoint>，主题 g1ra/action，载荷 380 字节 G1RA ACTION 帧

订阅：
  - ZMQ <state_endpoint>，主题 g1ra/state，载荷 292 字节 G1RA STATE 帧
  - 解析后重建 unitree_hg/LowState 发到 <lowstate_out_topic>，
    并用 robot_state 生成 /robot_state 字符串，供 policy_node 使用。

按键：默认从 /wirelesscontroller 读遥控器并写进重建的 LowState，
让 policy_node 的 A/B/Y 逻辑保持不变。桥自身仍用机器人的遥控器驱动状态机。
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from unitree_hg.msg import LowState
from unitree_go.msg import WirelessController

import zmq

from g1_remote_action import protocol

N_JOINTS = protocol.N_JOINTS


class G1raStateReceiver:
    """后台线程收 STATE 帧，只保留最新一帧。"""

    def __init__(self, endpoint: str, mode: str = "connect", recv_timeout_ms: int = 20) -> None:
        self._endpoint = endpoint
        self._mode = mode
        self._recv_timeout_ms = int(recv_timeout_ms)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SUBSCRIBE, protocol.TOPIC_STATE)
        if mode == "bind":
            self._sock.bind(endpoint)
        else:
            self._sock.connect(endpoint)

        self._latest: Optional[protocol.StateFrame] = None
        self._latest_recv_time: float = 0.0
        self.received = 0
        self.decode_errors = 0
        self.last_error = ""
        self._stop = False

        import threading

        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="g1ra-state-recv", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop:
            try:
                raw = self._sock.recv()
            except zmq.Again:
                continue
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"recv failed: {exc}"
                time.sleep(0.05)
                continue
            try:
                payload = protocol.strip_topic(raw, protocol.TOPIC_STATE)
                frame = protocol.decode_state(payload)
            except protocol.ProtocolError as exc:
                self.decode_errors += 1
                self.last_error = str(exc)
                continue
            with self._lock:
                self._latest = frame
                self._latest_recv_time = time.monotonic()
                self.received += 1

    def latest(self) -> Optional[protocol.StateFrame]:
        with self._lock:
            return self._latest

    def age(self) -> float:
        with self._lock:
            if self._latest is None:
                return float("inf")
            return time.monotonic() - self._latest_recv_time

    def close(self) -> None:
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self._sock.close(0)
        except Exception:  # noqa: BLE001
            pass


class G1raAdapterNode(Node):
    """把 HoloMotion 的输出转成 G1RA ACTION，并把 STATE 回灌成 LowState。"""

    def __init__(self, args) -> None:
        super().__init__("g1ra_adapter")
        self.args = args

        self.declare_parameter("action_topic", args.action_topic)
        self.declare_parameter("kps_topic", args.kps_topic)
        self.declare_parameter("kds_topic", args.kds_topic)
        self.declare_parameter("lowstate_out_topic", args.lowstate_out_topic)
        self.declare_parameter("robot_state_topic", args.robot_state_topic)

        self._kp = np.zeros(N_JOINTS, dtype=np.float32)
        self._kd = np.zeros(N_JOINTS, dtype=np.float32)
        self._kp_ok = False
        self._kd_ok = False
        self._gains_wait_since: Optional[float] = None
        self._gains_hint_shown = False

        self._seq = 0
        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 1)
        self._pub.setsockopt(zmq.CONFLATE, 1)
        self._pub.setsockopt(zmq.LINGER, 0)
        if args.action_mode == "bind":
            self._pub.bind(args.action_endpoint)
        else:
            self._pub.connect(args.action_endpoint)

        self._state_rx = G1raStateReceiver(
            endpoint=args.state_endpoint,
            mode=args.state_mode,
        )
        self._state_rx.start()

        self._lowstate_pub = self.create_publisher(LowState, args.lowstate_out_topic, 10)
        self._robot_state_pub = self.create_publisher(String, args.robot_state_topic, 10)

        # 遥控器原始包优先用 STATE 帧里的扩展段；机器人本机的 /wirelesscontroller 仅作后备
        self._wireless_raw: Optional[bytes] = None
        self.create_subscription(WirelessController, args.wireless_topic, self._on_wireless, 10)

        self.create_subscription(Float32MultiArray, args.action_topic, self._on_action, 10)
        self.create_subscription(Float32MultiArray, args.kps_topic, self._on_kps, 10)
        self.create_subscription(Float32MultiArray, args.kds_topic, self._on_kds, 10)

        self._state_timer = self.create_timer(1.0 / max(args.state_repulish_hz, 1.0), self._republish_state)
        self._status_timer = self.create_timer(1.0, self._log_status)

        self.get_logger().info(
            "g1ra_adapter 启动: action=%s(%s) state=%s(%s) lowstate_out=%s robot_state=%s"
            % (
                args.action_endpoint,
                args.action_mode,
                args.state_endpoint,
                args.state_mode,
                args.lowstate_out_topic,
                args.robot_state_topic,
            )
        )

    # ---------------- ROS 回调 ----------------

    def _on_wireless(self, msg: WirelessController) -> None:
        keys = int(msg.keys)
        raw = keys.to_bytes(2, "little")
        for value in (msg.lx, msg.rx, msg.ry):
            raw += np.float32(value).tobytes()
        raw += np.float32(0.0).tobytes() * 2
        raw += np.float32(msg.ly).tobytes()
        self._wireless_raw = raw.ljust(40, b"\x00")[:40]

    def _on_kps(self, msg: Float32MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.float32)
        if arr.shape == (N_JOINTS,):
            self._kp = arr
            self._kp_ok = True

    def _on_kds(self, msg: Float32MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.float32)
        if arr.shape == (N_JOINTS,):
            self._kd = arr
            self._kd_ok = True

    def _on_action(self, msg: Float32MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.float32)
        if arr.shape != (N_JOINTS,) or not np.all(np.isfinite(arr)):
            self.get_logger().warn(f"action 长度或数值非法: shape={arr.shape}")
            return
        if not self._kp_ok or not self._kd_ok:
            now = time.monotonic()
            if self._gains_wait_since is None:
                self._gains_wait_since = now
            if not self._gains_hint_shown and now - self._gains_wait_since > 5.0:
                self._gains_hint_shown = True
                self.get_logger().error(
                    "/humanoid/kps 与 /humanoid/kds 5 s 内未收到，动作帧全部丢弃。"
                    "常见原因：policy_node 没启动；或启动了全身模式但缺少 "
                    "velocity_tracking_model（policy_node 强制双策略，会直接启动失败）。"
                    "当前阶段只支持上半身模式：由 webcam_teleop（仅加载 motion 模型）提供 "
                    "/humanoid/action 与 /humanoid/kps|kds。"
                )
            self.get_logger().warn("kps/kds 尚未就绪，丢帧", throttle_duration_sec=1.0)
            return
        self._gains_wait_since = None
        self._seq += 1
        payload = protocol.encode_action(
            seq=self._seq,
            stamp_ns=time.monotonic_ns(),
            q=arr,
            kp=self._kp,
            kd=self._kd,
            flags=protocol.FLAG_ENABLE,
        )
        self._pub.send(protocol.TOPIC_ACTION + payload)

    # ---------------- STATE 回灌 ----------------

    def _republish_state(self) -> None:
        frame = self._state_rx.latest()
        if frame is None:
            return
        # stamp_ns 是本机单调时钟，与远端不同时钟源；估龄用接收线程的到达时刻。
        if self._state_rx.age() > float(self.args.state_max_age_ms) / 1000.0:
            return
        msg = LowState()
        for i in range(N_JOINTS):
            msg.motor_state[i].q = float(frame.q[i])
            msg.motor_state[i].dq = float(frame.dq[i])
        msg.imu_state.quaternion = [float(v) for v in frame.quat_wxyz]
        msg.imu_state.gyroscope = [float(v) for v in frame.gyro]
        wireless = frame.wireless if frame.wireless else self._wireless_raw
        if wireless:
            msg.wireless_remote = bytes(wireless).ljust(40, b"\x00")[:40]
        self._lowstate_pub.publish(msg)

        state_msg = String()
        state_msg.data = protocol.robot_state_name(frame.robot_state)
        self._robot_state_pub.publish(state_msg)

    def _log_status(self) -> None:
        age = self._state_rx.age()
        age_ms = None if age == float("inf") else age * 1000.0
        self.get_logger().info(
            "[g1ra] seq=%d rx=%d err=%d state_age_ms=%s kp_ok=%s kd_ok=%s"
            % (
                self._seq,
                self._state_rx.received,
                self._state_rx.decode_errors,
                "None" if age_ms is None else f"{age_ms:.1f}",
                self._kp_ok,
                self._kd_ok,
            )
        )

    def destroy_node(self) -> None:
        self._state_rx.close()
        try:
            self._pub.close(0)
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HoloMotion <-> g1-remote-action 适配器")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5610")
    parser.add_argument("--action-mode", default="connect", choices=("connect", "bind"))
    parser.add_argument("--state-endpoint", default="tcp://127.0.0.1:5611")
    parser.add_argument("--state-mode", default="connect", choices=("connect", "bind"))
    parser.add_argument("--action-topic", default="/humanoid/action")
    parser.add_argument("--kps-topic", default="/humanoid/kps")
    parser.add_argument("--kds-topic", default="/humanoid/kds")
    parser.add_argument("--lowstate-out-topic", default="/g1ra/lowstate")
    parser.add_argument("--robot-state-topic", default="/g1ra/robot_state")
    parser.add_argument("--wireless-topic", default="/wirelesscontroller")
    parser.add_argument("--state-repulish-hz", type=float, default=100.0)
    parser.add_argument("--state-max-age-ms", type=float, default=200.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    rclpy.init(args=None)
    node = G1raAdapterNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
