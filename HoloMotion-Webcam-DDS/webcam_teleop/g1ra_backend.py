"""G1RA 后端：把策略输出打成 G1RA ACTION 帧，经 ZMQ 发给机器人侧的 g1-remote-action 桥。

与 dds_backend 的区别：这里**不推进** MuJoCo 物理，机器人本身就是被执行对象。
本进程每个策略周期只做三件事：

1. 从 ev.target_dof_pos_by_name 取 29 维目标角（按桥的关节顺序重排）；
2. 把目标角 + 模型自带的 kp/kd 打成 380 字节 ACTION 帧发出去；
3. 用桥回传的 STATE 覆盖 ev.d 的 qpos/qvel/IMU，供官方观测构造继续使用。

桥负责关节限位、力矩和断流看门狗。本后端另做实测姿态接管、策略目标限速、
可选的下肢锁定和反馈断流停发；下肢锁定在本次接管的实测角，而非模型默认站姿。

帧格式见 g1-remote-action 的 docs/interface.md；编解码直接复用
g1_remote_action.protocol，避免出现第二份实现。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import zmq
import zmq.utils.monitor

LOWER_BODY_KEYS = ("hip_", "knee", "ankle")


def _import_protocol():
    """复用 g1-remote-action 的协议模块；找不到时按 G1RA_HOME 兜底。"""
    try:
        from g1_remote_action import protocol  # noqa: PLC0415

        return protocol
    except ImportError:
        home = Path(os.environ.get("G1RA_HOME", Path.home() / "dev" / "g1-remote-action"))
        src = home / "src"
        if src.is_dir():
            sys.path.insert(0, str(src))
        try:
            from g1_remote_action import protocol  # noqa: PLC0415

            return protocol
        except ImportError as exc:  # pragma: no cover - 环境问题
            raise RuntimeError(
                "g1ra 后端需要 g1-remote-action 的协议模块："
                f"请把 G1RA_HOME 指到 g1-remote-action 目录（当前尝试 {src}），"
                f"或安装该包（pip install -e <g1-remote-action>）。原始错误：{exc}"
            ) from exc


class G1raBackend:
    """把 ev 的目标角转发给机器人，并把机器人实测状态回灌给 ev。"""

    def __init__(
        self,
        ev,
        action_endpoint: str = "tcp://127.0.0.1:5610",
        state_endpoint: str = "tcp://127.0.0.1:5611",
        action_mode: str = "connect",
        state_mode: str = "connect",
        lock_lower_body: bool = True,
        state_timeout_s: float = 0.5,
        control_mode: str = 'full_body',
    ) -> None:
        self.protocol = _import_protocol()
        if not hasattr(self.protocol, 'PHASE_ALIGNMENT'):
            raise RuntimeError('请更新 G1RA_HOME 中的协议模块以支持重力前馈阶段标识')
        self.ev = ev
        self.state_timeout_s = float(state_timeout_s)
        self.lock_lower_body = bool(lock_lower_body)
        if control_mode not in ('arm_sdk', 'full_body'):
            raise ValueError('未知 G1RA 控制接口模式')
        self.control_mode = control_mode
        self.feedback_source = 'robot'
        self.simulation = None

        # 关节顺序取自部署配置的 complete_dof_order（与桥的 joint_spec.names 已逐项核对一致）
        self.names = [str(name) for name in ev.dof_names_ref_motion]
        if len(self.names) != self.protocol.N_JOINTS:
            raise ValueError(
                f"需要 {self.protocol.N_JOINTS} 个关节，实际 {len(self.names)} 个；"
                "G1RA 协议只支持 G1 29DOF"
            )

        self.qpos_idx = [int(ev.m.joint(name).qposadr[0]) for name in self.names]
        self.qvel_idx = [int(ev.m.joint(name).dofadr[0]) for name in self.names]
        mu = [ev.mjcf_dof_names.index(name) for name in self.names]
        # ACTION 携带模型增益；机器人 SDK 桥实际采用其自身 spec.kp/kd。
        self.kp = np.asarray([ev.kps_mu[i] for i in mu], dtype=np.float32)
        self.kd = np.asarray([ev.kds_mu[i] for i in mu], dtype=np.float32)
        # 站姿取自参考序列第 0 帧（即部署配置的 default_joint_angles）
        self.stance = np.asarray(ev.ref_dof_pos[0], dtype=np.float32).copy()
        self.lower_idx = [
            i for i, name in enumerate(self.names)
            if any(key in name for key in LOWER_BODY_KEYS)
        ]

        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 1)
        self._pub.setsockopt(zmq.CONFLATE, 1)
        self._pub.setsockopt(zmq.LINGER, 0)

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.RCVHWM, 1)
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.setsockopt(zmq.SUBSCRIBE, self.protocol.TOPIC_STATE)

        # monitor 必须在 connect/bind 之前挂，否则首次连接事件会漏掉。
        # 它只反映 TCP 层的建立/断开，不参与实际收发。
        self.action_connected = False
        self.state_connected = False
        try:
            self._pub_mon = self._pub.get_monitor_socket()
            self._sub_mon = self._sub.get_monitor_socket()
        except Exception:  # noqa: BLE001
            self._pub_mon = self._sub_mon = None

        if action_mode == "bind":
            self._pub.bind(action_endpoint)
        else:
            self._pub.connect(action_endpoint)
        if state_mode == "bind":
            self._sub.bind(state_endpoint)
        else:
            self._sub.connect(state_endpoint)

        self.seq = 0
        self.sent = 0
        self.pairs = 0
        self.state_seq = 0
        self.state_age_ms = None
        self.robot_state = self.protocol.ROBOT_STATE_UNKNOWN
        self._state = None
        self._state_time = 0.0
        self._state_is_valid = False
        self._stale_warned = False
        self._recv_errors = 0
        self.needs_sync = True
        self._last_command = None
        self._reference_q = None
        self._reference_first = True
        self._align_reference = True

        print(
            f"[g1ra] action={action_endpoint}({action_mode}) state={state_endpoint}({state_mode}) "
            f"lock_lower_body={self.lock_lower_body} joints={len(self.names)}",
            flush=True,
        )

    # ---------------- 收状态 ----------------

    def _poll_states(self) -> None:
        """非阻塞取走所有排队帧，只保留最新一帧。"""
        while True:
            try:
                raw = self._sub.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception:  # noqa: BLE001
                self._recv_errors += 1
                break
            try:
                payload = self.protocol.strip_topic(raw, self.protocol.TOPIC_STATE)
                frame = self.protocol.decode_state(payload)
            except Exception:  # noqa: BLE001
                self._recv_errors += 1
                continue
            self._state = frame
            self._state_time = time.monotonic()
            self.state_seq = int(frame.seq)
            self.robot_state = int(frame.robot_state)
            # 桥还没拿到 /lowstate 时 q/dq 全 0，不能拿去喂观测
            self._state_is_valid = bool(
                int(frame.bridge_flags) & self.protocol.STATE_FLAG_LOWSTATE_VALID
            )

    def _state_fresh(self) -> bool:
        if self._state is None:
            return False
        return (time.monotonic() - self._state_time) <= self.state_timeout_s

    def _drain_monitor(self, sock, attr: str) -> None:
        """非阻塞取走 monitor 事件，只关心连接建立与断开。"""
        if sock is None:
            return
        while True:
            try:
                # 必须 NOBLOCK：这里跑在 50 Hz 控制线程上，阻塞会直接卡住仿真。
                event = zmq.utils.monitor.recv_monitor_message(sock, zmq.NOBLOCK)
            except Exception:  # noqa: BLE001 - Again/关闭/无事件都直接收工
                return
            kind = event.get("event")
            if kind == zmq.EVENT_CONNECTED:
                setattr(self, attr, True)
            elif kind in (zmq.EVENT_DISCONNECTED, zmq.EVENT_CLOSED):
                setattr(self, attr, False)

    def poll_link(self) -> None:
        """链路心跳：不推进控制时也持续刷新连接标志与状态帧年龄。

        暂停状态下也要能看到「机器人还连着吗、状态还在来吗」，
        所以这里不依赖 step()，由控制线程每轮调用。
        """
        self._poll_states()
        self.state_age_ms = (
            None if self._state is None else (time.monotonic() - self._state_time) * 1000.0
        )
        self._drain_monitor(self._pub_mon, "action_connected")
        self._drain_monitor(self._sub_mon, "state_connected")

    def _apply_state(self) -> None:
        """把机器人实测 q/dq/IMU 写回 mjData，让官方观测构造拿到真实状态。"""
        frame = self._state
        ev = self.ev
        d = ev.d
        d.qpos[self.qpos_idx] = np.asarray(frame.q, dtype=np.float64)
        d.qvel[self.qvel_idx] = np.asarray(frame.dq, dtype=np.float64)
        # 没有实测基座平移速度，不沿用纯仿真阶段的残留速度。
        d.qvel[:3] = 0.0
        # MuJoCo 自由关节的四元数与 STATE 一样是 (w, x, y, z)，可直接拷贝
        d.qpos[3:7] = tuple(float(v) for v in frame.quat_wxyz)
        # STATE gyro 是 IMU/基座局部坐标系的角速度（rad/s）；
        # MuJoCo free joint 的转动 qvel 同样使用基座局部坐标系，无需旋转到世界系。
        d.qvel[3:6] = np.asarray(frame.gyro, dtype=np.float64)
        mujoco.mj_forward(ev.m, d)
        self.pairs += 1

    # ---------------- 发动作 ----------------

    def _target(self) -> np.ndarray:
        q = np.empty(self.protocol.N_JOINTS, dtype=np.float32)
        for i, name in enumerate(self.names):
            q[i] = self.ev.target_dof_pos_by_name.get(name, float(self.stance[i]))
        if self.lock_lower_body and self.lower_idx:
            q[self.lower_idx] = self.stance[self.lower_idx]
        return q

    def _publish(self, q: np.ndarray, phase: int = 0) -> None:
        self.seq += 1
        self.sent += 1
        payload = self.protocol.encode_action(
            seq=self.seq,
            stamp_ns=time.monotonic_ns(),
            q=q,
            kp=self.kp,
            kd=self.kd,
            flags=self.protocol.FLAG_ENABLE,
            phase=phase,
        )
        self._pub.send(self.protocol.TOPIC_ACTION + payload)

    def sync_hold(self) -> bool:
        """只接受新鲜实测状态，重建策略起点；失败时不发送默认站姿。"""
        self.poll_link()
        if not self._state_fresh() or not self._state_is_valid:
            return False
        frame = self._state
        values = np.r_[frame.q, frame.dq, frame.quat_wxyz, frame.gyro]
        if not np.isfinite(values).all() or np.linalg.norm(frame.quat_wxyz) < 1e-6:
            return False
        ev = self.ev
        simulation = getattr(self, 'simulation', None)
        if simulation is not None:
            # Seed local physics from measured pose once at takeover. Do not
            # overwrite it with delayed measurements on subsequent policy steps.
            ev.d = simulation.measured_data
        # 清除 KV、步号、动作延迟等历史，再用实测状态覆盖 teleport 结果。
        ev.reset_state_teleport()
        ev.obs_builder.reset()
        self._apply_state()
        if simulation is not None:
            simulation.seed(ev.d)
            simulation.select_feedback(self.feedback_source)
        q = np.asarray(frame.q, dtype=np.float32).copy()
        self.stance = q.copy()
        self._last_command = q.copy()
        self._reference_q = q.copy()
        self._reference_first = True
        self._align_reference = True
        ev.target_dof_pos_by_name = dict(zip(self.names, map(float, q)))
        # last_action 与当前目标一致，避免把仿真动作历史带入第一帧观测。
        q_by_name = ev.target_dof_pos_by_name
        ev.actions_onnx = (
            np.asarray([q_by_name[n] for n in ev.dof_names_onnx], dtype=np.float32)
            - ev.default_angles_onnx
        ) / ev.action_scale_onnx
        gate = getattr(self, '_simulation_gate', None)
        if gate is not None:
            gate.reset()
        self.needs_sync = False
        self._publish(q, phase=self.protocol.PHASE_HOLD)
        return True

    def prepare_policy_step(self, force_measured=False) -> bool:
        """先更新实测状态，再构造策略观测；断流后需要重新开始同步。"""
        self.poll_link()
        if self.needs_sync or not self._state_fresh() or not self._state_is_valid:
            self.needs_sync = True
            return False
        values = np.r_[self._state.q, self._state.dq, self._state.quat_wxyz, self._state.gyro]
        if not np.isfinite(values).all() or np.linalg.norm(self._state.quat_wxyz) < 1e-6:
            self.needs_sync = True
            return False
        simulation = getattr(self, 'simulation', None)
        source = 'robot' if force_measured else getattr(self, 'feedback_source', 'robot')
        if simulation is not None:
            simulation.select_feedback(source)
        if source == 'robot':
            self._apply_state()
        return True

    def measured_snapshot(self, reference):
        """Independent render-only copy, without altering policy feedback data."""
        if not self._state_fresh() or not self._state_is_valid:
            return None
        frame = self._state
        if not np.isfinite(np.r_[frame.q, frame.dq, frame.quat_wxyz, frame.gyro]).all():
            return None
        base = self.simulation.measured_data if self.simulation is not None else self.ev.d
        qpos, qvel = base.qpos.copy(), np.zeros_like(base.qvel)
        qpos[self.qpos_idx] = frame.q
        qpos[3:7] = frame.quat_wxyz
        qvel[self.qvel_idx] = frame.dq
        qvel[3:6] = frame.gyro
        return qpos, qvel, np.asarray(reference).copy()

    def advance_simulation(self, command):
        simulation = getattr(self, 'simulation', None)
        if simulation is None:
            return
        try:
            simulation.advance(self.names, command, self.control_mode)
            if self.feedback_source == 'simulation':
                # Last-action observations must describe the targets actually
                # applied to local physics, including the arm output slew cap.
                ev = self.ev
                ev.actions_onnx = (
                    np.asarray([simulation.last_targets[n] for n in ev.dof_names_onnx], dtype=np.float32)
                    - ev.default_angles_onnx
                ) / ev.action_scale_onnx
        except ValueError:
            # A render-only preview failure does not change real-robot policy
            # feedback. If simulation feeds the policy, stop issuing commands.
            if self.feedback_source == 'simulation':
                self.needs_sync = True
                raise

    def reference_target(self, q: np.ndarray, max_step: float) -> np.ndarray:
        """第一帧参考为实测角，此后按参考限速靠近人体目标。"""
        if self.needs_sync or self._reference_q is None:
            raise ValueError('请先同步机器人当前姿态')
        if self._reference_first:
            self._reference_first = False
        else:
            self._reference_q += np.clip(q - self._reference_q, -max_step, max_step)
        if self.lock_lower_body:
            self._reference_q[self.lower_idx] = self.stance[self.lower_idx]
        return self._reference_q.copy()

    def send_alignment(self, q: np.ndarray) -> None:
        """仅供冻结目标对齐；不调用策略，仍要求新鲜状态和有界命令。"""
        if self.needs_sync or not self._state_fresh() or not self._state_is_valid:
            raise ValueError('实机反馈过期，对齐已停止')
        q = np.asarray(q, dtype=np.float32)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise ValueError('对齐指令无效')
        if np.max(np.abs(q - self._last_command)) > .00501:
            raise ValueError('对齐指令变化过快，已停止')
        self._publish(q, phase=self.protocol.PHASE_ALIGNMENT)
        self._last_command = q.copy()

    def alignment_control_ready(self) -> bool:
        """arm_sdk 的 3/4 是流状态；full_body 的 3 才是控制状态机 POLICY。"""
        if self.control_mode == 'arm_sdk':
            if self.robot_state not in (self.protocol.ROBOT_STATE_POLICY, self.protocol.ROBOT_STATE_POLICY_HOLD):
                raise ValueError('手臂桥未就绪或已急停，对齐已停止')
            return True
        return self.robot_state == self.protocol.ROBOT_STATE_POLICY

    def reset(self) -> None:
        """仅清除接管起点；页面复位不再向实机发送默认站姿。"""
        self.needs_sync = True
        self._last_command = self._reference_q = None

    # ---------------- 被 ev._apply_control 调用 ----------------

    def step(self, sleep: bool = False) -> None:
        self.poll_link()

        if self.needs_sync or not self._state_fresh() or not self._state_is_valid:
            self.needs_sync = True
            # 机器人状态断流/未就绪：停发，让桥的 100 ms 看门狗把机器人收进 POLICY_HOLD
            if not self._stale_warned:
                self._stale_warned = True
                print(
                    "[g1ra] 未收到有效的机器人状态（bridge_flags 缺 LOWSTATE_VALID 或超过 "
                    f"{self.state_timeout_s:.1f}s 未更新），已停止下发动作",
                    flush=True,
                )
            return

        self._stale_warned = False
        # 每个策略周期最多 0.02 rad（50 Hz 时 1 rad/s），长延迟不扩大单次跳变量。
        q = self._last_command + np.clip(self._target() - self._last_command, -.02, .02)
        if not np.isfinite(q).all():
            self.needs_sync = True
            raise ValueError('策略目标包含非有限值，已停止下发')
        # Validate local physics BEFORE publishing any policy frame.
        from simulation_gate import SimulationGate
        gate = getattr(self, '_simulation_gate', None)
        if gate is None:
            gate = self._simulation_gate = SimulationGate()
        simulation = getattr(self, 'simulation', None)
        if simulation is None:
            gate.reset()
            self.needs_sync = True
            raise ValueError('缺少本地仿真，已禁止发送策略动作')
        try:
            simulation.advance(self.names, q, self.control_mode)
        except ValueError:
            gate.reset()
            self.needs_sync = True
            raise
        indices = [i for i, n in enumerate(self.names)
                   if self.control_mode == 'full_body'
                   or any(part in n for part in ('shoulder', 'elbow', 'wrist'))]
        limits = np.asarray([self.ev.m.joint(self.names[i]).range for i in indices])
        passed = gate.check(
            simulation.data.qpos[np.asarray(self.qpos_idx)[indices]],
            simulation.data.qvel[np.asarray(self.qvel_idx)[indices]],
            q[indices], limits, float(simulation.data.qpos[2]))
        self.simulation_gate_status = gate.reason
        if self.feedback_source == 'simulation':
            ev = self.ev
            ev.actions_onnx = (
                np.asarray([simulation.last_targets[n] for n in ev.dof_names_onnx], dtype=np.float32)
                - ev.default_angles_onnx
            ) / ev.action_scale_onnx
        if not passed:
            # Discard unvalidated output. Robot-side timeout/hold remains active.
            return
        # Recheck feedback after potentially expensive simulation work.
        self.poll_link()
        if not self._state_fresh() or not self._state_is_valid:
            gate.reset()
            self.needs_sync = True
            raise ValueError('仿真验证后实机反馈过期，未发送动作')
        self._publish(q, phase=self.protocol.PHASE_POLICY)
        self._last_command = q

    def close(self) -> None:
        for sock in (self._pub_mon, self._sub_mon, self._pub, self._sub):
            try:
                sock.close(0)
            except Exception:  # noqa: BLE001
                pass
