"""g1-remote-action 线协议。

本模块**不依赖 ROS**，可以在任意 Python 3.8+ 环境（含 conda 环境）里单独
做编解码、单测和联调，方便远端后端按同一份规范实现。

两种帧：

    ACTION  远端 -> 本机 : 29 关节绝对目标角 + kp/kd
    STATE   本机 -> 远端 : 29 关节实测 q/dq + IMU + 状态码

全部采用小端定长二进制 + CRC32 校验，具体字节布局见 docs/interface.md。
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

MAGIC = b"G1RA"
PROTOCOL_VERSION = 1

MSG_TYPE_ACTION = 1
MSG_TYPE_STATE = 2

N_JOINTS = 29
N_QUAT = 4
N_GYRO = 3

# ZMQ 主题前缀（单帧消息 = 主题前缀 + 二进制载荷，便于 SUB 侧按前缀过滤）
TOPIC_ACTION = b"g1ra/action"
TOPIC_STATE = b"g1ra/state"

# ACTION.flags 位定义
FLAG_ENABLE = 0x0001  # 远端明确授权本机下发动作；未置位视为"暂停"
FLAGS_KNOWN_MASK = FLAG_ENABLE

# ACTION 原 reserved uint16：0 保持旧发送端的零前馈行为。
PHASE_UNSPECIFIED = 0
PHASE_ALIGNMENT = 1
PHASE_POLICY = 2
PHASE_HOLD = 3
PHASES = (0, 1, 2, 3)

# STATE.bridge_flags 位定义
STATE_FLAG_LOWSTATE_VALID = 0x0001  # 本机已收到 /lowstate
STATE_FLAG_PUBLISHING = 0x0002      # 本机当前正在下发动作
STATE_FLAG_STALE = 0x0004           # 本机已判定动作流超时
STATE_FLAG_REJECTING = 0x0008       # 本机正在拒绝远端帧（见状态接口）
STATE_FLAG_GRAVITY_SUPPORTED = 0x0010
STATE_FLAG_GRAVITY_ENABLED = 0x0020

# 机器人状态码（与 deployment 的 /robot_state 字符串一一对应）
ROBOT_STATE_UNKNOWN = 0
ROBOT_STATE_ZERO_TORQUE = 1
ROBOT_STATE_MOVE_TO_DEFAULT = 2
ROBOT_STATE_POLICY = 3
ROBOT_STATE_POLICY_HOLD = 4
ROBOT_STATE_EMERGENCY_STOP = 5

ROBOT_STATE_BY_NAME = {
    "UNKNOWN": ROBOT_STATE_UNKNOWN,
    "ZERO_TORQUE": ROBOT_STATE_ZERO_TORQUE,
    "MOVE_TO_DEFAULT": ROBOT_STATE_MOVE_TO_DEFAULT,
    "POLICY": ROBOT_STATE_POLICY,
    "POLICY_HOLD": ROBOT_STATE_POLICY_HOLD,
    "EMERGENCY_STOP": ROBOT_STATE_EMERGENCY_STOP,
}
ROBOT_STATE_NAME_BY_CODE = {v: k for k, v in ROBOT_STATE_BY_NAME.items()}

_ACTION_STRUCT = struct.Struct("<4sHHQqHH29f29f29fI")
_STATE_STRUCT = struct.Struct("<4sHHQq29f29f4f3fHHI")
# STATE v2：在 gyro 与 robot_state 之间插入 40 字节遥控器原始包，帧长 332。
# 解码按长度区分：292 = v1 布局（wireless 为空），332 = v2 布局。
_STATE_WIRELESS_STRUCT = struct.Struct("<4sHHQq29f29f4f3f40sHHI")
EXT_WIRELESS_LEN = 40

ACTION_FRAME_SIZE = _ACTION_STRUCT.size                # 380
STATE_FRAME_SIZE = _STATE_STRUCT.size                  # 292（无遥控器段）
STATE_WIRELESS_FRAME_SIZE = _STATE_WIRELESS_STRUCT.size  # 334（带遥控器段）


class ProtocolError(ValueError):
    """帧格式/校验失败。"""


@dataclass
class ActionFrame:
    """远端下发的一帧动作。"""

    seq: int
    stamp_ns: int
    flags: int
    q: np.ndarray   # float32[29] 绝对目标关节角 (rad)
    kp: np.ndarray  # float32[29]
    kd: np.ndarray  # float32[29]
    phase: int = PHASE_UNSPECIFIED

    @property
    def enabled(self) -> bool:
        return bool(self.flags & FLAG_ENABLE)


@dataclass
class StateFrame:
    """本机回传的一帧机器人状态。"""

    seq: int
    stamp_ns: int
    q: np.ndarray        # float32[29] rad
    dq: np.ndarray       # float32[29] rad/s
    quat_wxyz: np.ndarray
    gyro: np.ndarray
    robot_state: int
    bridge_flags: int
    # 宇树遥控器 40 字节原始包（SDK 后端转发 rt/lowstate.wireless_remote）。
    # 远端可借此还原 A/B/Y 等按键，驱动 policy_node 的模式切换。
    wireless: bytes = b""

    @property
    def lowstate_valid(self) -> bool:
        return bool(self.bridge_flags & STATE_FLAG_LOWSTATE_VALID)


def _as_vec(values: Sequence[float], size: int, name: str, allow_nonfinite: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (size,):
        raise ProtocolError(f"{name} must have shape ({size},), got {arr.shape}")
    if not allow_nonfinite and not np.all(np.isfinite(arr)):
        raise ProtocolError(f"{name} contains non-finite values")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _finish(struct_obj: struct.Struct, fields: tuple) -> bytes:
    """打包并在尾部追加 CRC32（覆盖除 CRC 本身以外的全部字节）。"""
    raw = struct_obj.pack(*fields, 0)
    crc = zlib.crc32(raw[:-4]) & 0xFFFFFFFF
    return raw[:-4] + struct.pack("<I", crc)


def _check_header(payload: bytes, expect_size: int, expect_type: int, struct_obj: struct.Struct) -> None:
    if len(payload) != expect_size:
        raise ProtocolError(f"frame size mismatch: got {len(payload)}, expect {expect_size}")
    magic, version, msg_type = struct_obj.unpack_from(payload, 0)[:3]
    if magic != MAGIC:
        raise ProtocolError(f"bad magic: {magic!r}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")
    if msg_type != expect_type:
        raise ProtocolError(f"unexpected msg_type: {msg_type}, expect {expect_type}")
    crc_stored = struct.unpack_from("<I", payload, expect_size - 4)[0]
    crc_calc = zlib.crc32(payload[:-4]) & 0xFFFFFFFF
    if crc_stored != crc_calc:
        raise ProtocolError("crc32 mismatch")


def encode_action(
    seq: int,
    stamp_ns: int,
    q: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    flags: int = FLAG_ENABLE,
    allow_nonfinite: bool = False,
    phase: int = PHASE_UNSPECIFIED,
) -> bytes:
    """编码 ACTION 帧。

    allow_nonfinite 仅供测试注入异常帧，正常链路禁止使用。
    """
    if phase not in PHASES:
        raise ProtocolError(f"unknown action phase: {phase}")
    q_arr = _as_vec(q, N_JOINTS, "q", allow_nonfinite)
    kp_arr = _as_vec(kp, N_JOINTS, "kp", allow_nonfinite)
    kd_arr = _as_vec(kd, N_JOINTS, "kd", allow_nonfinite)
    return _finish(
        _ACTION_STRUCT,
        (
            MAGIC,
            PROTOCOL_VERSION,
            MSG_TYPE_ACTION,
            int(seq) & 0xFFFFFFFFFFFFFFFF,
            int(stamp_ns),
            int(flags) & 0xFFFF,
            int(phase),
            *q_arr.tolist(),
            *kp_arr.tolist(),
            *kd_arr.tolist(),
        ),
    )


def decode_action(payload: bytes) -> ActionFrame:
    _check_header(payload, ACTION_FRAME_SIZE, MSG_TYPE_ACTION, _ACTION_STRUCT)
    fields = _ACTION_STRUCT.unpack(payload)
    _, _, _, seq, stamp_ns, flags, phase, *rest = fields
    if phase not in PHASES:
        raise ProtocolError(f"unknown action phase: {phase}")
    q = np.asarray(rest[0:29], dtype=np.float32)
    kp = np.asarray(rest[29:58], dtype=np.float32)
    kd = np.asarray(rest[58:87], dtype=np.float32)
    return ActionFrame(seq=seq, stamp_ns=stamp_ns, flags=flags, q=q, kp=kp, kd=kd, phase=phase)


def encode_state(
    seq: int,
    stamp_ns: int,
    q: Sequence[float],
    dq: Sequence[float],
    quat_wxyz: Sequence[float],
    gyro: Sequence[float],
    robot_state: int = ROBOT_STATE_UNKNOWN,
    bridge_flags: int = 0,
    wireless: Optional[bytes] = None,
) -> bytes:
    q_arr = _as_vec(q, N_JOINTS, "q", False)
    dq_arr = _as_vec(dq, N_JOINTS, "dq", False)
    quat_arr = _as_vec(quat_wxyz, N_QUAT, "quat_wxyz", False)
    gyro_arr = _as_vec(gyro, N_GYRO, "gyro", False)
    wl = bytes(wireless) if wireless else b""
    if wireless:
        if len(wl) != EXT_WIRELESS_LEN:
            raise ProtocolError(f"wireless must be {EXT_WIRELESS_LEN} bytes, got {len(wl)}")
        return _finish(
            _STATE_WIRELESS_STRUCT,
            (
                MAGIC,
                PROTOCOL_VERSION,
                MSG_TYPE_STATE,
                int(seq) & 0xFFFFFFFFFFFFFFFF,
                int(stamp_ns),
                *q_arr.tolist(),
                *dq_arr.tolist(),
                *quat_arr.tolist(),
                *gyro_arr.tolist(),
                wl,
                int(robot_state) & 0xFFFF,
                int(bridge_flags) & 0xFFFF,
            ),
        )
    return _finish(
        _STATE_STRUCT,
        (
            MAGIC,
            PROTOCOL_VERSION,
            MSG_TYPE_STATE,
            int(seq) & 0xFFFFFFFFFFFFFFFF,
            int(stamp_ns),
            *q_arr.tolist(),
            *dq_arr.tolist(),
            *quat_arr.tolist(),
            *gyro_arr.tolist(),
            int(robot_state) & 0xFFFF,
            int(bridge_flags) & 0xFFFF,
        ),
    )


def decode_state(payload: bytes) -> StateFrame:
    wireless = b""
    if len(payload) == STATE_WIRELESS_FRAME_SIZE:
        _check_header(payload, STATE_WIRELESS_FRAME_SIZE, MSG_TYPE_STATE, _STATE_WIRELESS_STRUCT)
        fields = _STATE_WIRELESS_STRUCT.unpack(payload)
        _, _, _, seq, stamp_ns, *rest = fields
        q = np.asarray(rest[0:29], dtype=np.float32)
        dq = np.asarray(rest[29:58], dtype=np.float32)
        quat = np.asarray(rest[58:62], dtype=np.float32)
        gyro = np.asarray(rest[62:65], dtype=np.float32)
        wireless = bytes(rest[65])
        robot_state, bridge_flags = rest[66], rest[67]
    else:
        _check_header(payload, STATE_FRAME_SIZE, MSG_TYPE_STATE, _STATE_STRUCT)
        fields = _STATE_STRUCT.unpack(payload)
        _, _, _, seq, stamp_ns, *rest = fields
        q = np.asarray(rest[0:29], dtype=np.float32)
        dq = np.asarray(rest[29:58], dtype=np.float32)
        quat = np.asarray(rest[58:62], dtype=np.float32)
        gyro = np.asarray(rest[62:65], dtype=np.float32)
        robot_state, bridge_flags = rest[65], rest[66]
    return StateFrame(
        seq=seq,
        stamp_ns=stamp_ns,
        q=q,
        dq=dq,
        quat_wxyz=quat,
        gyro=gyro,
        robot_state=robot_state,
        bridge_flags=bridge_flags,
        wireless=wireless,
    )


def strip_topic(payload: bytes, topic: bytes) -> bytes:
    """去掉 ZMQ 主题前缀，返回纯帧。"""
    if payload.startswith(topic):
        return payload[len(topic):]
    raise ProtocolError(f"topic prefix mismatch, expect {topic!r}")


def robot_state_name(code: int) -> str:
    return ROBOT_STATE_NAME_BY_CODE.get(int(code), "UNKNOWN")


def frame_size_report() -> dict:
    """便于自检和文档对齐。"""
    return {
        "action_frame_size": ACTION_FRAME_SIZE,
        "state_frame_size": STATE_FRAME_SIZE,
        "magic": MAGIC.decode(),
        "version": PROTOCOL_VERSION,
        "n_joints": N_JOINTS,
    }
