"""真机位姿源：从机器人侧 joint_relay 的 HTTP 接口取 29 关节角与 IMU。

取数方式抄 camview/robot_side/joint_relay.py：
机器人上跑一个只读 relay（订阅 ``rt/lowstate``，HTTP 暴露 ``/q``），
本机按关节名把实测角映射进 MuJoCo 的 qpos，并用 IMU rpy 还原基座姿态。

几个必须知道的边界：
* G1 的 lowstate **不含基座位置**，所以 qpos[0:3] 保持模型默认，不做航位推算；
* IMU yaw 会缓慢漂移，只适合显示，不能当定位用；
* 这里只服务渲染，不参与策略控制。

接口返回示例（relay 的 /q）::

    {"ok": true, "seq": 815064, "age_ms": 1.0, "hz": 611.6,
     "mode_machine": 5, "names": ["left_hip_pitch_joint", ...],
     "q": [29 floats], "dq": [29 floats], "imu_rpy": [r, p, y]}
"""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.request

import mujoco
import numpy as np


def rpy_to_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """固定轴 R = Rz·Ry·Rx（与 camview / URDF 约定一致），返回 (w, x, y, z)。"""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


class HttpPoseSource:
    """后台定频拉取 /q，缓存成可直接写入 mjData 的 qpos / qvel。"""

    def __init__(self, model: mujoco.MjModel, base: str, hz: float = 50.0, timeout: float = 0.5):
        self.model = model
        self.url = base.rstrip("/") + "/q"
        self.period = 1.0 / max(1.0, hz)
        self.timeout = timeout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        self._qpos = np.array(model.qpos0, dtype=np.float64)
        self._qvel = np.zeros(model.nv, dtype=np.float64)
        self._ok = False
        self._error = None
        self._stamp = 0.0
        self._seq = -1
        self._hz = 0.0
        self._mode = None
        self._matched = 0

        # 关节名 -> (qpos 地址, dof 地址)；free joint 跳过，基座单独处理。
        self._addr: dict[str, tuple[int, int]] = {}
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not name or model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            self._addr[name] = (int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i]))

    # ---------------- 线程 ----------------

    def start(self) -> "HttpPoseSource":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="pose-http", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("ok"):
                    self._ingest(payload)
                else:
                    with self._lock:
                        self._ok = False
                        self._error = payload.get("note") or "relay 未收到 rt/lowstate"
            except Exception as exc:  # noqa: BLE001 - 断网/超时都不该拖垮渲染
                with self._lock:
                    self._ok = False
                    self._error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(max(0.0, self.period - (time.monotonic() - started)))

    def _ingest(self, payload: dict) -> None:
        age = float(payload.get('age_ms', 0.))
        if not np.isfinite(age) or age < 0 or age > 500:
            raise ValueError('实机位姿接口反馈过期')
        names = payload.get("names") or []
        q = payload.get("q") or []
        dq = payload.get("dq") or []
        rpy = payload.get("imu_rpy") or [0.0, 0.0, 0.0]

        with self._lock:
            qpos = self._qpos.copy()
            qvel = self._qvel.copy()

        matched = 0
        for i, name in enumerate(names):
            slot = self._addr.get(name)
            if slot is None or i >= len(q):
                continue
            qpos[slot[0]] = float(q[i])
            if i < len(dq):
                qvel[slot[1]] = float(dq[i])
            matched += 1

        # 基座姿态用 IMU 还原；位置没有反馈，保持 qpos[0:3] 不变。
        qpos[3:7] = rpy_to_wxyz(*(float(v) for v in rpy))
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError('实机位姿包含无效数值')

        with self._lock:
            self._qpos, self._qvel = qpos, qvel
            self._ok = matched > 0
            self._matched = matched
            self._error = None if matched else "关节名与 MuJoCo 模型全不匹配"
            if int(payload.get('seq', -1)) != self._seq or not self._stamp:
                self._stamp = time.monotonic()
            self._seq = int(payload.get("seq", -1))
            self._hz = float(payload.get("hz") or 0.0)
            self._mode = payload.get("mode_machine")

    # ---------------- 读取 ----------------

    def snapshot(self):
        """返回 (qpos, qvel)；尚无有效数据时返回 None。"""
        with self._lock:
            if not self._ok or time.monotonic() - self._stamp > .5:
                return None
            return self._qpos.copy(), self._qvel.copy()

    def status(self) -> dict:
        with self._lock:
            age = None if not self._stamp else round((time.monotonic() - self._stamp) * 1000.0, 1)
            return {
                "ok": self._ok and age is not None and age <= 500,
                "age_ms": age,
                "hz": round(self._hz, 1),
                "seq": self._seq,
                "mode_machine": self._mode,
                "matched_joints": self._matched,
                "total_joints": len(self._addr),
                "error": self._error,
            }
