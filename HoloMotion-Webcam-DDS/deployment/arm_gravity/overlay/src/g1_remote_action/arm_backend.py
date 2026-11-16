#!/usr/bin/env python3
"""arm_sdk 后端（默认）：只接管上肢 + 锁腰，官方运控继续管腿和平衡。

与 sdk_backend（全身接管 `rt/lowcmd`）的区别：

  - **不 release 官方运控**，官方 `ai` 模式保持运行，腿和平衡归它；
  - 只写 `rt/arm_sdk`，接管 15..28 共 14 个臂关节；
  - 腰 12..14 在使能瞬间读取实测角并固定，全程高刚度保持
    （不锁腰的话，上身在重力下会栽倒）；
  - `motor_cmd[29].q` 是 arm_sdk 的使能权重：0 -> 1 渐入、1 -> 0 渐出；
  - 远端 ACTION 帧仍是 29 维，但桥只取 `q[15:29]`，腰和腿的维度被忽略。

信号处理参考官方遥操 `xr_teleoperate`（`teleop/robot_control/robot_arm.py`）：

  - 增益分级：肩/肘 `kp_high`，腕 `kp_wrist`（裸 kp=40 撑不住手臂自重，会下垂乱晃）；
  - 关节目标加权滑动平均（默认 [0.4, 0.3, 0.2, 0.1]，输入不变则不推进窗口）；
  - 可选死区保持：目标变化小于 `deadband_rad` 就不更新，切断静止抖动；
  - 断流默认**不重置**爬坡基准（`reset_on_stale: false`），保持最后目标。

全部参数在 `config/bridge.yaml` 的 `arm:` 段，可用 `--set arm.kp_high=250` 覆盖。

用法：
    python -m g1_remote_action.arm_backend --interface eth0
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from typing import Optional

import numpy as np

from . import protocol
from .config import apply_overrides, load_config
from .core import ActionProcessor
from .limits import build_spec, motor_lookup
from .gravity_compensation import GravityCompensator, GravityModels, settings
from .sdk_backend import Measured, extract_measured, init_channel
from .transport import ActionReceiver, StatePublisher, make_state_frame, now_ns

ARM = list(range(15, 29))                      # 左臂 15..21，右臂 22..28
ARM_HIGH = [15, 16, 17, 18, 22, 23, 24, 25]    # 肩 pitch/roll/yaw + 肘
ARM_WRIST = [19, 20, 21, 26, 27, 28]           # 腕 roll/pitch/yaw
WAIST = [12, 13, 14]                           # yaw / roll / pitch
WEIGHT_IDX = 29                                # arm_sdk 使能权重关节
WEIGHT_RAMP_DT = 0.02

# 默认值参考 xr_teleoperate/teleop/robot_control/robot_arm.py:66-73
ARM_DEFAULTS = {
    "kp_high": 300.0,          # 肩/肘刚度
    "kd_high": 3.0,
    "kp_wrist": 40.0,          # 腕刚度
    "kd_wrist": 1.5,
    "waist_kp": 300.0,         # 腰锁刚度（xr 用 kp_high 锁腰）
    "waist_kd": 3.0,
    "filter_weights": [0.4, 0.3, 0.2, 0.1],   # 加权滑动平均；只填 1 个 1.0 即关闭
    "deadband_rad": 0.0,       # 目标变化小于该值就保持不动，0 = 关闭
    "reset_on_stale": False,   # 断流后是否从实测角重新爬坡
    "weight_ramp_sec": 1.2,    # arm_sdk 权重 0->1 渐入时长
    "max_slew_rad_s": None,    # None = 用 control.max_slew_rad_s
}


def arm_config(cfg) -> dict:
    """取出 arm 段配置，缺省值补齐（未知键忽略，向前兼容）。"""
    merged = dict(ARM_DEFAULTS)
    raw = cfg.get("arm") or {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in merged and value is not None:
                merged[key] = value
    weights = merged["filter_weights"]
    if not isinstance(weights, (list, tuple)) or len(weights) == 0:
        raise SystemExit("arm.filter_weights 必须是非空数组")
    return merged


class WeightedMovingFilter:
    """加权滑动平均，照搬 xr_teleoperate/teleop/utils/weighted_moving_filter.py。

    输入与上一拍完全相同时不推进窗口 —— 静止时滤波器不会自行爬行。
    """

    def __init__(self, weights, data_size: int = len(ARM)) -> None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size == 0 or float(w.sum()) <= 0.0:
            raise ValueError("filter_weights 必须是非空且和为正的数组")
        self._weights = w / float(w.sum())
        self._window = int(self._weights.size)
        self._size = int(data_size)
        self._queue = []
        self._filtered = np.zeros(self._size, dtype=np.float32)

    @property
    def filtered(self) -> np.ndarray:
        return self._filtered

    @property
    def window(self) -> int:
        return self._window

    def update(self, data) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64).reshape(-1)
        if arr.size != self._size:
            raise ValueError(f"filter 输入维度应为 {self._size}，实际 {arr.size}")
        if self._queue and np.array_equal(arr, self._queue[-1]):
            return self._filtered
        self._queue.append(arr)
        if len(self._queue) > self._window:
            self._queue.pop(0)
        if len(self._queue) < self._window:
            self._filtered = arr.astype(np.float32)
            return self._filtered
        stack = np.stack(self._queue)
        self._filtered = (stack * self._weights[:, None]).sum(axis=0).astype(np.float32)
        return self._filtered


class ArmLink:
    """`rt/arm_sdk` 发布 + `rt/lowstate` 订阅（不发 `rt/lowcmd`）。"""

    def __init__(self, spec, acfg: dict, gravity_enabled=False) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.spec = spec
        self._crc = CRC()
        self._msg = unitree_hg_msg_dds__LowCmd_()
        for slot in self._msg.motor_cmd:
            slot.mode = 0

        self._kp_high = float(acfg["kp_high"])
        self._kd_high = float(acfg["kd_high"])
        self._kp_wrist = float(acfg["kp_wrist"])
        self._kd_wrist = float(acfg["kd_wrist"])
        self._waist_kp = float(acfg["waist_kp"])
        self._waist_kd = float(acfg["waist_kd"])

        self._latest = None
        self._body_sample = (None, float('-inf'))
        self._hand_samples = [(None, float('-inf')), (None, float('-inf'))]
        self._hand_subs = []
        self.rx_count = 0
        self.tx_count = 0
        self.last_error = ""

        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        if gravity_enabled:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
            for side, index in (('left', 0), ('right', 1)):
                sub = ChannelSubscriber(f'rt/dex3/{side}/state', HandState_)
                sub.Init(lambda msg, i=index: self._on_hand(msg, i), 10)
                self._hand_subs.append(sub)

    def _on_state(self, msg) -> None:
        self._body_sample = (msg, time.monotonic())
        self._latest = msg
        self.rx_count += 1

    def latest(self):
        return self._latest

    def _on_hand(self, msg, index):
        try:
            q = np.array([m.q for m in msg.motor_state], dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError('invalid hand state')
            self._hand_samples[index] = (q, time.monotonic())
        except Exception:
            self._hand_samples[index] = (None, float('-inf'))

    def gravity_sample(self, now):
        msg, stamp = self._body_sample
        left, right = self._hand_samples
        try:
            body = np.array([msg.motor_state[j].q for j in range(29)])
            quat = np.asarray(msg.imu_state.quaternion, dtype=float)
            fingers = np.r_[left[0], right[0]] if left[0] is not None and right[0] is not None else None
            return body, fingers, quat, now-stamp, max(now-left[1], now-right[1])
        except Exception:
            return None, None, None, float('inf'), float('inf')

    def publish(self, arm_target, waist_lock, weight: float, tau=None) -> None:
        """arm_target: 14 维（左 7 + 右 7）；waist_lock: 3 维。"""
        msg = self._msg
        for idx, j in enumerate(ARM):
            c = msg.motor_cmd[j]
            c.mode = 1
            c.q = float(arm_target[idx])
            c.dq = 0.0
            c.tau = 0.0 if tau is None else float(tau[idx])
            if j in ARM_WRIST:
                c.kp, c.kd = self._kp_wrist, self._kd_wrist
            else:
                c.kp, c.kd = self._kp_high, self._kd_high
        for idx, j in enumerate(WAIST):
            c = msg.motor_cmd[j]
            c.mode = 1
            c.q = float(waist_lock[idx])
            c.dq = 0.0
            c.tau = 0.0
            c.kp, c.kd = self._waist_kp, self._waist_kd
        if self._latest is not None:
            msg.mode_machine = int(self._latest.mode_machine)
        msg.motor_cmd[WEIGHT_IDX].q = float(weight)
        try:
            msg.crc = self._crc.Crc(msg)
            self._pub.Write(msg)
            self.tx_count += 1
        except Exception as exc:  # noqa: BLE001 - 写失败不应拖垮控制循环
            self.last_error = f"arm_sdk write failed: {exc}"

    def disable(self) -> None:
        """权重归零并把臂/腰交还给官方运控（mode=0）。"""
        for j in ARM + WAIST:
            self._msg.motor_cmd[j].mode = 0
        self._msg.motor_cmd[WEIGHT_IDX].q = 0.0
        try:
            self._msg.crc = self._crc.Crc(self._msg)
            self._pub.Write(self._msg)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"arm_sdk disable failed: {exc}"

    def close(self) -> None:
        for sock in (self._pub, self._sub, *self._hand_subs):
            try:
                sock.Close()
            except Exception:  # noqa: BLE001
                pass


class ArmBridge:
    def __init__(self, cfg, spec, args, link: ArmLink, gravity) -> None:
        ctl = cfg["control"]
        tr = cfg["transport"]
        ros = cfg["ros"]

        self.cfg = cfg
        self.spec = spec
        self.args = args
        self.link = link
        self.gravity = gravity
        self._gravity_time = time.monotonic()
        self.acfg = arm_config(cfg)
        self.dry_run = bool(args.dry_run)
        self.control_hz = float(args.control_hz or 250.0)
        self.period = 1.0 / self.control_hz
        self.restore_sec = float(getattr(args, "restore_sec", 2.0))
        self.deadband = float(self.acfg["deadband_rad"])

        self.processor = ActionProcessor(
            spec,
            timeout_s=float(ctl["action_timeout_ms"]) / 1000.0,
            min_valid_frames=int(ctl["min_valid_frames"]),
            require_enable=bool(ctl["require_enable_flag"]),
            safety_scale=float(ctl["safety_scale"]),
            reset_on_stale=bool(self.acfg["reset_on_stale"]),
            reject_on_kp_kd_mismatch=bool(ctl["reject_on_kp_kd_mismatch"]),
        )
        self._filter = WeightedMovingFilter(self.acfg["filter_weights"])
        self._motor_idx = motor_lookup(spec)
        self._receiver = ActionReceiver(
            endpoint=str(tr["action"]["endpoint"]),
            mode=str(tr["action"].get("mode", "bind")),
            topic=str(tr["topics"]["action"]).encode("utf-8"),
            recv_timeout_ms=int(tr["action"].get("recv_timeout_ms", 20)),
        )
        self._state_pub: Optional[StatePublisher] = None
        if bool(ros.get("publish_state", True)):
            self._state_pub = StatePublisher(
                endpoint=str(tr["state"]["endpoint"]),
                mode=str(tr["state"].get("mode", "bind")),
                topic=str(tr["topics"]["state"]).encode("utf-8"),
                send_hwm=int(tr["state"].get("send_hwm", 2)),
            )

        self._start_arm: Optional[np.ndarray] = None
        self._waist_lock: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None
        self._last_raw: Optional[np.ndarray] = None
        self._state_seq = 0
        self._last_state_pub = 0.0
        self._last_status = 0.0
        self._t0 = time.monotonic()
        self._enabled = False
        self._deadband_skips = 0

    # ---------------- 使能 / 释放 ----------------

    def _enable(self, measured: Measured) -> None:
        self._start_arm = np.array([measured.q[j] for j in ARM], dtype=np.float32)
        self._waist_lock = np.array([measured.q[j] for j in WAIST], dtype=np.float32)
        self._target = self._start_arm.copy()
        self._last_raw = self._start_arm.copy()
        if self.dry_run:
            print("[init] dry-run：不使能 arm_sdk，也不下发任何指令")
            return
        ramp_sec = max(float(self.acfg["weight_ramp_sec"]), 0.0)
        steps = max(int(ramp_sec / WEIGHT_RAMP_DT), 1)
        for k in range(1, steps + 1):
            self.link.publish(self._target, self._waist_lock, k / steps)
            time.sleep(WEIGHT_RAMP_DT)
        self._enabled = True
        waist_deg = np.degrees(self._waist_lock)
        print(
            f"[init] arm_sdk 已使能（权重 1.0）；腰锁定 yaw/roll/pitch = "
            f"{float(waist_deg[0]):.1f}/{float(waist_deg[1]):.1f}/{float(waist_deg[2]):.1f}°"
        )
        print(
            f"[init] 增益：肩/肘 kp={self.acfg['kp_high']:.2f} kd={self.acfg['kd_high']:.2f} | "
            f"腕 kp={self.acfg['kp_wrist']:.2f} kd={self.acfg['kd_wrist']:.2f} | "
            f"腰 kp={self.acfg['waist_kp']:.1f} kd={self.acfg['waist_kd']:.1f}"
        )
        print(
            f"[init] 平滑：滑动平均窗口={self._filter.window} "
            f"权重={list(np.round(self.acfg['filter_weights'], 3))} | "
            f"死区={self.deadband * 1000:.1f} mrad | "
            f"断流重置={self.acfg['reset_on_stale']}"
        )

    def _release(self) -> None:
        if not self._enabled or self._target is None or self._waist_lock is None:
            return
        print(f"[run] 手臂回起始位（{self.restore_sec:.1f}s）...")
        q0 = self._target.copy()
        base = self._start_arm if self._start_arm is not None else q0
        n = max(int(self.restore_sec / self.period), 10)
        for k in range(1, n + 1):
            r = 0.5 * (1.0 - np.cos(np.pi * k / n))
            tau = self._gravity_update(0, False)
            self.link.publish(q0 + (base - q0) * r, self._waist_lock, 1.0, tau)
            time.sleep(self.period)
        # Finish fading residual feedforward before releasing arm_sdk weight.
        while np.max(np.abs(self.gravity.tau)) > 1e-8:
            self.link.publish(base, self._waist_lock, 1.0, self._gravity_update(0, False))
            time.sleep(self.period)
        print("[run] 释放 arm_sdk 权重 1.0 -> 0.0 ...")
        steps = max(int(float(self.acfg["weight_ramp_sec"]) / WEIGHT_RAMP_DT), 1)
        for w in np.linspace(1.0, 0.0, num=steps):
            self.link.publish(base, self._waist_lock, float(w))
            time.sleep(WEIGHT_RAMP_DT)
        self.link.disable()
        self._enabled = False
        print("[run] 已释放：臂和腰交还官方运控")

    # ---------------- 状态回传 ----------------

    def _publish_state(self, measured: Measured, following: bool, stale: bool) -> None:
        if self._state_pub is None:
            return
        self._state_seq += 1
        flags = protocol.STATE_FLAG_LOWSTATE_VALID | protocol.STATE_FLAG_GRAVITY_SUPPORTED
        if self.gravity.enabled:
            flags |= protocol.STATE_FLAG_GRAVITY_ENABLED
        if following:
            flags |= protocol.STATE_FLAG_PUBLISHING
        if stale:
            flags |= protocol.STATE_FLAG_STALE
        code = protocol.ROBOT_STATE_POLICY if following else protocol.ROBOT_STATE_POLICY_HOLD
        self._state_pub.send(
            make_state_frame(
                seq=self._state_seq,
                stamp_ns=now_ns(),
                q=measured.q,
                dq=measured.dq,
                quat_wxyz=measured.quat_wxyz,
                gyro=measured.gyro,
                robot_state=code,
                bridge_flags=flags,
                wireless=measured.wireless,
            )
        )

    def _status(self, measured: Measured, following: bool) -> None:
        info = {
            "t": round(time.monotonic() - self._t0, 1),
            "mode": "arm_sdk",
            "state": "ARM_FOLLOW" if following else "ARM_HOLD",
            "reason": self.processor.last_reason,
            "seq": self.processor.last_seq,
            "rx": self._receiver.received,
            "rx_err": self._receiver.decode_errors,
            "tx": self.link.tx_count,
            "lowstate_rx": self.link.rx_count,
            "skip_deadband": self._deadband_skips,
            "gravity": self.gravity.status(),
        }
        if self._waist_lock is not None:
            info["waist_deg"] = [round(float(v), 1) for v in np.degrees(self._waist_lock)]
        counters = self.processor.active_counters()
        if counters:
            info["counters"] = counters
        if self.link.last_error:
            info["link_error"] = self.link.last_error
        if self._receiver.last_error:
            info["rx_last_error"] = self._receiver.last_error
        print("[status] " + json.dumps(info, ensure_ascii=False), flush=True)

    # ---------------- 主循环 ----------------

    def _gravity_update(self, phase, active):
        now = time.monotonic()
        dt = now - self._gravity_time
        self._gravity_time = now
        sample = self.link.gravity_sample(now)
        return self.gravity.update(phase, active, *sample, now, dt)

    def _update_target(self, arm_target: np.ndarray) -> None:
        """滑动平均 + 死区保持（参考官方遥操的静止不动机制）。"""
        if self.deadband > 0.0 and self._last_raw is not None:
            if float(np.max(np.abs(arm_target - self._last_raw))) < self.deadband:
                self._deadband_skips += 1
                return          # 变化量在死区内：保持现有目标，滤波窗口也不推进
        self._last_raw = arm_target.copy()
        self._target = self._filter.update(arm_target).copy()

    def run(self) -> int:
        self._receiver.start()

        deadline = time.monotonic() + 5.0
        while self.link.latest() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.link.latest() is None:
            print(
                "[init] 5 s 内没有收到 rt/lowstate：检查网卡/网线/机器人是否上电。未做任何接管。",
                file=sys.stderr,
            )
            return 4

        measured = extract_measured(self.link.latest(), self._motor_idx)
        print(f"[init] 已收到 rt/lowstate（mode_machine={measured.mode_machine}）")

        print(
            f"[run] control={self.control_hz:.0f}Hz dry_run={self.dry_run} "
            f"action={self.cfg['transport']['action']['endpoint']}"
        )
        print("[run] 模式=arm_sdk：只接管 15..28 上肢，腰 12..14 锁定；腿和平衡归官方运控")

        self._enable(measured)

        try:
            while True:
                loop_start = time.monotonic()
                ls = self.link.latest()
                following = False
                phase = protocol.PHASE_UNSPECIFIED
                if ls is not None:
                    measured = extract_measured(ls, self._motor_idx)
                    self.processor.note_measured(measured.q)
                    result = self.processor.process(
                        self._receiver.latest(), time.monotonic(), self.period
                    )
                    if result.target is not None:
                        arm_target = np.asarray(result.target, dtype=np.float32)[ARM]
                        self._update_target(arm_target)
                        following = True
                        phase = result.phase

                tau = self._gravity_update(phase, following)
                if self._target is not None and self._waist_lock is not None and not self.dry_run:
                    self.link.publish(self._target, self._waist_lock, 1.0, tau)

                stale = self.processor.last_reason == "stale"
                now = time.monotonic()
                if now - self._last_state_pub >= 0.02:
                    self._last_state_pub = now
                    self._publish_state(measured, following, stale)
                if now - self._last_status >= 1.0:
                    self._last_status = now
                    self._status(measured, following)

                sleep_for = self.period - (time.monotonic() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("[run] 收到停止信号，执行臂回位并释放", flush=True)
            self._release()
        finally:
            self._receiver.close()
            if self._state_pub is not None:
                self._state_pub.close()
        return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="g1-remote-action arm_sdk 后端（只动上肢）")
    parser.add_argument("--config", default="config/bridge.yaml")
    parser.add_argument("--robot-config", default="")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--control-hz", type=float, default=0.0, help="0 表示 250 Hz")
    parser.add_argument("--dry-run", action="store_true", help="只计算不下发")
    parser.add_argument("--restore-sec", type=float, default=2.0, help="退出时手臂回位用时")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument('--gravity-preflight', action='store_true', help='离线模型检查，绝不初始化 DDS')
    parser.add_argument('--state-json', default='', help='preflight 使用的已保存 /q JSON')
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(
            "[init] arm_sdk 模式忽略参数: " + " ".join(unknown),
            file=sys.stderr,
            flush=True,
        )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    pairs = []
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式: {item}")
        key, value = item.split("=", 1)
        pairs.append((key.strip(), value.strip()))
    apply_overrides(cfg, pairs)

    robot_config = args.robot_config or None
    acfg = arm_config(cfg)
    slew = acfg["max_slew_rad_s"]
    slew = float(slew) if slew is not None else float(cfg["control"]["max_slew_rad_s"])

    spec = build_spec(
        cfg,
        robot_config_path=robot_config,
        max_slew_rad_s=slew,
    )

    gcfg = settings(cfg, args.config)
    if args.gravity_preflight:
        from .gravity_compensation import FINGERS
        models = GravityModels(gcfg, spec.names)
        if not args.state_json:
            raise SystemExit('--gravity-preflight requires --state-json')
        from pathlib import Path
        state = json.loads(Path(args.state_json).read_text())
        body = dict(zip(state['names'], state['q']))
        hands = {name: value for hand in state['hands'].values()
                 for name, value in zip(hand['names'], hand['q'])}
        # Saved relay exposes imu_rpy; construct its wxyz quaternion explicitly.
        roll, pitch, yaw = np.asarray(state['imu_rpy']) / 2
        cr, cp, cy = np.cos([roll, pitch, yaw])
        sr, sp, sy = np.sin([roll, pitch, yaw])
        quat = [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
        full, delta = models.compute([body[n] for n in spec.names], [hands[n] for n in FINGERS], quat)
        print(json.dumps(dict(no_robot_io=True, configured_enabled=gcfg['enabled'],
                              names=spec.names[15:29], full_nm=full.tolist(), delta_nm=delta.tolist())))
        return 0
    # Validate dependencies/models before initializing any DDS command publisher.
    gravity = GravityCompensator(gcfg, spec)

    try:
        import unitree_sdk2py  # noqa: F401
    except ImportError:
        print(
            "unitree_sdk2py 不可用：请激活带 SDK 的环境（例如 conda tv）。",
            file=sys.stderr,
        )
        return 2

    def _on_sigterm(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:
        pass  # 非主线程注册会失败，忽略

    print(
        f"[init] 模式=arm_sdk（只动手臂，官方运控保持运行）"
        f" 网卡={args.interface} domain={args.domain_id} 限速={slew:.1f} rad/s"
    )
    init_channel(args.interface, args.domain_id)

    link = ArmLink(spec, acfg, gravity.enabled)
    bridge = ArmBridge(cfg, spec, args, link, gravity)
    try:
        return bridge.run()
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
