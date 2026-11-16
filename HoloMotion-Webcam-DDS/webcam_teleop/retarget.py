"""Webcam-to-G1 arm/torso adapter using Mink; not HoloRetarget."""
import time
import numpy as np
import mujoco

import retarget_config
from tracking_filter import PoseFilter, JointHold
from mink_ik import MinkArmIK


# 将向量归一化为方向；两关键点重合时方向不可靠，拒绝该帧。
def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-5:
        raise ValueError('关键点重合，请站远一点并让上半身完整入镜')
    return v / n


# 将人体坐标转换成 G1 参考动作；不负责神经网络推理或真实物理控制。
class WebcamRetarget:
    def __init__(self, model, joint_names, body_names, defaults, config=None):
        # 所有可调参数集中在这里：校准门限 calibration / 映射限幅 mapping，
        # 见同目录 retarget_config.yaml。
        self.cfg = config if config is not None else retarget_config.load()
        self.calib = self.cfg['calibration']
        self.map = self.cfg['mapping']
        self.tracking = self.cfg['tracking']
        self.pose_filter = PoseFilter(self.tracking, self.calib['min_visibility'])
        self.joint_hold = JointHold(self.tracking)
        self.m = model
        # 这是参考姿态专用 MjData，与仿真机器人的实际 MjData 分开。
        self.d = mujoco.MjData(model)
        self.names = list(joint_names)
        self.body_ids = [model.body(n).id for n in body_names]
        # qpos 包含浮动根部，关节角不一定从索引 0 开始，按名称查询地址。
        self.qadr = np.array([model.joint(n).qposadr[0] for n in joint_names])
        self.nominal = np.asarray(defaults, dtype=float).copy()
        self.q = self.nominal.copy()
        self.root = np.array([0., 0., .8])
        self.basis = None
        self.last_update = None
        # 最近一次校准试算出的最大偏差（弧度），供前端展示校准质量。
        self.calib_delta_rad = None
        self.fk(self.q)
        ankles = [model.body(s + '_ankle_roll_link').id for s in ('left', 'right')]
        # 调整初始根部高度，让最低脚踝中心位于约 0.035 m；并非脚底接触求解。
        self.root[2] += .035 - min(self.d.xpos[b, 2] for b in ankles)
        self.fk(self.q)
        self.initial_pos = self.d.xpos[self.body_ids].copy()
        # MuJoCo xquat 是 w,x,y,z；参考协议改成 x,y,z,w。
        self.initial_quat = self.d.xquat[self.body_ids][:, [1, 2, 3, 0]].copy()
        # 每条手臂求解肩 pitch/roll/yaw 和肘共 4 个角，腕关节保持默认参考。
        self.arms = {}
        for side in ('left', 'right'):
            names = [side + '_' + s + '_joint' for s in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow')]
            indices = [self.names.index(n) for n in names]
            ids = [model.joint(n).id for n in names]
            shoulder, elbow, wrist = [model.body(side + '_' + n + '_link').id for n in ('shoulder_roll', 'elbow', 'wrist_yaw')]
            upper = np.linalg.norm(self.d.xpos[elbow] - self.d.xpos[shoulder])
            lower = np.linalg.norm(self.d.xpos[wrist] - self.d.xpos[elbow])
            self.arms[side] = dict(indices=indices, vids=[model.jnt_dofadr[i] for i in ids],
                bounds=model.jnt_range[ids].copy(), shoulder=shoulder, elbow=elbow, wrist=wrist,
                lengths=(upper, lower))
        self.ik = MinkArmIK(model, self.arms, self.map)
        self.reset_tracking()

    def reset_tracking(self):
        self.pose_filter.reset()
        self.joint_hold.reset(self.q)
        self._sample_id = None
        self._target_q = self.q.copy()

    # 正向运动学：给定关节角，求所有机器人刚体的位置和朝向，不推进时间。
    def fk(self, q):
        self.d.qpos[:3] = self.root
        self.d.qpos[3:7] = [1, 0, 0, 0]
        self.d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, self.d)

    # MediaPipe 索引：11/12肩，13/14肘，15/16腕，23/24髋，左右按人体自身。
    def validate(self, landmarks):
        p = np.asarray(landmarks, dtype=float)
        if p.shape != (33, 4) or not np.isfinite(p).all():
            raise ValueError('需要 33 个有限的人体世界坐标和可见度')
        floor = self.calib['min_visibility']
        if min(p[i, 3] for i in (11, 12, 13, 14, 15, 16, 23, 24)) < floor:
            raise ValueError(f'肩、肘、手腕、髋部需要清晰可见（可见度低于 {floor:.2f}）')
        return p[:, :3]

    # 从站姿建立前/左/上三个正交方向，将摄像头坐标转换到机器人参考坐标。
    # strict=True 时额外做静态姿态门限：update() 走关键点绝对方向映射，校准时人若
    # 已抬臂、叉腰、驼背，这些偏差会被原样放大成关节目标，表现就是「机器人乱动」。
    # strict=False 用于纯仿真：只求骨架与机器人对上，不拦姿态。
    def calibrate(self, landmarks, strict=True):
        p = self.validate(landmarks)
        c = self.calib
        up = unit((p[11] + p[12] - p[23] - p[24]) / 2)
        left = unit(p[11] - p[12])
        forward = unit(np.cross(left, up))
        left = unit(np.cross(up, forward))

        if strict:
            self._check_static_pose(p, up)

        self.basis = np.stack([forward, left, up])
        self.q = self.nominal.copy()
        self.last_update = None
        self.reset_tracking()
        if not strict:
            self.calib_delta_rad = None
            return 0.0

        # 最后一道闸：按当前姿态真算一遍目标，看会不会立刻甩离站姿。
        trial = self._solve_target(p)
        # 只比较决定手臂整体指向的肩部与腰部。肘部存在人机固有差异
        # （自然人直臂 vs 机器人训练站姿微弯肘 ~34°），计入会让正常站姿也超限。
        guarded = [i for i, n in enumerate(self.names) if 'shoulder' in n or 'waist' in n]
        worst = float(np.abs(trial[guarded] - self.nominal[guarded]).max())
        if worst > c['max_nominal_delta_rad']:
            self.basis = None
            raise ValueError(
                f'初始姿态让目标偏离机器人站姿 {np.degrees(worst):.0f}°'
                f'（上限 {np.degrees(c["max_nominal_delta_rad"]):.0f}°），'
                '请站直、双手自然下垂后重新校准')
        self.calib_delta_rad = worst
        return worst

    # 严格校准的静态姿态门限；不通过直接抛 ValueError。
    def _check_static_pose(self, p, up):
        c = self.calib
        # 躯干相对世界竖直线的倾角；取绝对值，不依赖 MediaPipe 的 y 轴朝向。
        tilt = np.degrees(np.arccos(np.clip(abs(up @ np.array([0., 1., 0.])), -1., 1.)))
        if tilt > c['max_torso_tilt_deg']:
            raise ValueError(f'躯干倾斜 {tilt:.0f}°（上限 {c["max_torso_tilt_deg"]:.0f}°），请站直再校准')

        down = -up
        devs = []
        for side, (s, e, w) in (('左', (11, 13, 15)), ('右', (12, 14, 16))):
            upper = unit(p[e] - p[s])
            fore = unit(p[w] - p[e])
            dev = np.degrees(np.arccos(np.clip(upper @ down, -1., 1.)))
            bend = np.degrees(np.arccos(np.clip(fore @ upper, -1., 1.)))
            devs.append(dev)
            if dev > c['max_upper_arm_dev_deg']:
                raise ValueError(
                    f'{side}臂偏离下垂 {dev:.0f}°（上限 {c["max_upper_arm_dev_deg"]:.0f}°），请双手自然垂下')
            if bend > c['max_elbow_bend_deg']:
                raise ValueError(
                    f'{side}肘弯曲 {bend:.0f}°（上限 {c["max_elbow_bend_deg"]:.0f}°），请把手臂伸直')
        if abs(devs[0] - devs[1]) > c['max_arm_asymmetry_deg']:
            raise ValueError(
                f'左右手臂不对称（差 {abs(devs[0] - devs[1]):.0f}°），请站正并让两臂对称下垂')

    # 输入 (33,4)；返回 q:(29,), pos:(30,3), quat:(30,4)。角度单位为弧度。
    def update(self, landmarks, dt=.02, sample_id=None, timestamp=None):
        if self.basis is None:
            raise ValueError('请先点击校准站姿')
        if sample_id is None:
            # Frozen-target calibration/offline callers deliberately refine one pose.
            target = self._solve_target(self.validate(landmarks))
        else:
            if self._sample_id is None:
                self._target_q = self.q.copy()
                self.joint_hold.reset(self.q)
            if sample_id != self._sample_id:
                timestamp = time.monotonic() if timestamp is None else timestamp
                if self.tracking['enabled']:
                    p = self.pose_filter.update(landmarks, timestamp)
                    if not self.pose_filter.held:
                        target = self._solve_target(p, prior=self._target_q)
                        self._target_q = self.joint_hold.update(target, timestamp)
                else:
                    self._target_q = self._solve_target(self.validate(landmarks), prior=self._target_q)
                self._sample_id = sample_id
            target = self._target_q
        m = self.map
        # Reference interpolation may run at 50 Hz; pose filtering and IK run only on new samples.
        alpha = 1. - (1. - m['smoothing']) ** (max(0., dt) / .02)
        self.q += np.clip((target - self.q) * alpha,
                          -m['max_rate_rad_s'] * dt, m['max_rate_rad_s'] * dt)
        self.fk(self.q)
        return self.q.copy(), self.d.xpos[self.body_ids].copy(), self.d.xquat[self.body_ids][:, [1, 2, 3, 0]].copy()

    # 按当前关键点算一帧关节目标（不含平滑与限速）；校准时也用它做试算。
    def _solve_target(self, p, prior=None):
        m = self.map
        target = self.q.copy() if prior is None else prior.copy()
        # 肩中心到髋中心方向用来估计躯干倾斜，腰部目标按配置限幅。
        torso = unit((p[11] + p[12] - p[23] - p[24]) / 2) @ self.basis.T
        ip = self.names.index('waist_pitch_joint')
        ir = self.names.index('waist_roll_joint')
        target[ip] = self.nominal[ip] + np.clip(np.arctan2(torso[0], torso[2]),
                                               -m['waist_pitch_limit_rad'], m['waist_pitch_limit_rad'])
        target[ir] = np.clip(-np.arctan2(torso[1], torso[2]),
                             -m['waist_roll_limit_rad'], m['waist_roll_limit_rad'])
        goals = {}
        self.fk(target)
        for side, points in [('left', (11, 13, 15)), ('right', (12, 14, 16))]:
            arm = self.arms[side]
            a, b, c = points
            # 只使用人体上臂/前臂方向，再乘机器人自身臂长，适应不同人体比例。
            u = unit(p[b] - p[a]) @ self.basis.T
            v = unit(p[c] - p[b]) @ self.basis.T
            shoulder = self.d.xpos[arm['shoulder']].copy()
            elbow_target = shoulder + u * arm['lengths'][0]
            wrist_target = elbow_target + v * arm['lengths'][1]
            goals[side, 'elbow'] = elbow_target
            goals[side, 'wrist'] = wrist_target
        solved = self.ik.solve(self.d.qpos, goals)
        # Only the eight shoulder/elbow joints are IK variables. Torso mapping,
        # wrists, legs and the floating root keep their existing ownership.
        for arm in self.arms.values():
            ix = arm['indices']
            target[ix] = solved[self.qadr[ix]]
        return target
