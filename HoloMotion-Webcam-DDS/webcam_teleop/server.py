"""Webcam landmark -> geometric G1 reference -> original HoloMotion ONNX -> MuJoCo."""
import argparse
from collections import deque
import json
import os
import threading
import tempfile
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# WSLg uses the Windows NVIDIA adapter through Mesa D3D12.
IS_WSL = 'microsoft' in Path('/proc/sys/kernel/osrelease').read_text().lower() if Path('/proc/sys/kernel/osrelease').exists() else False
os.environ.setdefault('MUJOCO_GL', 'glfw' if IS_WSL else 'egl')
if IS_WSL:
    os.environ.setdefault('GALLIUM_DRIVER', 'd3d12')
    os.environ.setdefault('MESA_D3D12_DEFAULT_ADAPTER_NAME', 'NVIDIA')
if os.environ['MUJOCO_GL'] == 'osmesa':
    raise RuntimeError('已禁止 OSMesa CPU 软件渲染；WSL 请使用 MUJOCO_GL=glfw')
import cv2
import mujoco
import numpy as np
from omegaconf import OmegaConf
from retarget import WebcamRetarget
from alignment import JointAlignment
from motion_diagnostics import MotionDiagnostics
from parallel_simulation import ParallelSimulation, choose_render_snapshot

# 项目根目录；网页静态文件放在 web 中，服务端不访问 Windows 摄像头。
ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / 'web'
# 线程共享状态：ready=模型就绪；running=允许跟随；calibrated=已校准。
# pose 保存最近一帧 (33,4) 人体关键点；last_pose 用服务端单调时钟记录接收时间。
STATE = dict(ready=False, stage='加载模型中', error=None, running=False,
             calibrated=False, last_pose=0., pose_time=0., pose_source_clock=False,
             sequence=-1, pose=None, reset=False,
             calibrate=False, steps=0, policy_hz=0., inference_ms=0., fallen=False,
             # robot_enabled=False 时是纯 MuJoCo 仿真：策略照跑，但不下发任何命令。
             robot_enabled=False,
             # auto_calibrated=True 表示基准是仿真模式自动建的，未经严格门限；
             # 切真机前必须重新手动校准。
             auto_calibrated=False, align=False, cancel_alignment=False,
             alignment_status='idle', alignment_error_deg=None, alignment_speed_deg_s=None,
             alignment_observed_settled=False, display_source='simulation',
             feedback_source='robot', feedback_pending=False,
             render_source=None, render_available=False)
# HTTP、仿真、渲染线程共享数据时用锁保护；耗时推理和渲染在锁外完成。
LOCK = threading.Lock()
# FRAME 是最近一张 JPEG；SNAPSHOT 是供渲染使用的机器人状态副本。
FRAME = b''
SNAPSHOT = None
SIM_SNAPSHOT = None
MEASURED_SNAPSHOT = None
MEASURED_STAMP = 0.
REFERENCE_EDGES = []
POSE_QUALITY_WINDOW = deque(maxlen=240)
VISIBILITY_FLOOR = .5
LANDMARK_LABELS = {11:'左肩',12:'右肩',13:'左肘',14:'右肘',15:'左腕',16:'右腕',23:'左髋',24:'右髋'}
STOP = threading.Event()
BACKEND = 'direct'
# g1ra 后端参数（仅在 --backend g1ra 时使用）：默认连本机 5610/5611，即机器人侧桥 bind 的端口。
ACTION_ENDPOINT = 'tcp://127.0.0.1:5610'
STATE_ENDPOINT = 'tcp://127.0.0.1:5611'
ACTION_MODE = 'connect'
STATE_MODE = 'connect'
LOCK_LOWER_BODY = True
STATE_TIMEOUT_S = 0.5
G1RA_CONTROL_MODE = os.environ.get('HOLOMOTION_G1RA_CONTROL_MODE', 'full_body')
# 真机位姿源（camview 的 joint_relay）：留空则画面继续用仿真状态。
POSE_ENDPOINT = ''
# 校准与映射参数 YAML；留空则用 webcam_teleop/retarget_config.yaml。
RETARGET_CONFIG_PATH = ''


# 初始化官方评估器：加载机器人 XML、ONNX 模型和观测构造器。
def make_evaluator():
    import retarget_config
    from holomotion.src.evaluation.eval_mujoco_sim2sim import MujocoEvaluator, process_config

    # 关闭官方离线状态记录；实时服务不累计保存用户动作。
    class LiveEvaluator(MujocoEvaluator):
        def _record_robot_states(self):
            pass  # Live mode has bounded buffers; no personal motion recording.

    os.chdir(ROOT)
    base = OmegaConf.load(ROOT / 'holomotion/config/evaluation/eval_mujoco_sim2sim.yaml')
    if 'defaults' in base:
        del base['defaults']
    # 初始化参考文件让官方评估器选择 motion_tracking；随后用实时参考覆盖它。
    override = OmegaConf.merge(base, OmegaConf.create(dict(
        ckpt_onnx_path=str(ROOT / 'checkpoints/HoloMotion_motion_tracking_model_v2.0.0/exported/model_10500.onnx'),
        robot_xml_path=str(ROOT / 'assets/robots/unitree/G1/29dof/scene_29dof.xml'),
        motion_npz_path=str(ROOT / 'data/first-demo/walk1_subject1_first10s.npz'), headless=True, record_video=False, use_gpu=True, require_cuda=True,
        dof_mode='29', dump_npzs=False, calc_per_clip_metrics=False,
        generate_report=False, camera_distance=2.8)))
    ev = LiveEvaluator(process_config(override))
    ev.setup()
    if ev.policy_session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('CUDA 推理初始化失败，禁止降级为 CPU 推理')
    ev.policy_session.disable_fallback()
    # 通过关节名称重排默认角度，不能假设模型顺序和仿真顺序相同。
    q = np.array([ev.default_angles_onnx[ev.dof_names_onnx.index(n)] for n in ev.dof_names_ref_motion])
    # 校准门限与映射限幅全部来自配置；写错直接抛出，不要带着离谱参数上机。
    retarget = WebcamRetarget(ev.m, ev.dof_names_ref_motion, ev.config.robot.body_names, q,
                              config=retarget_config.load(RETARGET_CONFIG_PATH or None))
    global REFERENCE_EDGES, VISIBILITY_FLOOR
    VISIBILITY_FLOOR = retarget.calib['min_visibility']
    ids = list(map(int, retarget.body_ids))
    lookup = {body: i for i, body in enumerate(ids)}
    REFERENCE_EDGES = []
    for body, i in lookup.items():
        parent = int(ev.m.body_parentid[body])
        while parent and parent not in lookup:
            parent = int(ev.m.body_parentid[parent])
        if parent in lookup and parent != body:
            REFERENCE_EDGES.append((lookup[parent], i))
    print('[retarget] Mink/DAQP：双臂肩肘位置 IK，原有关节限位，腕部保持默认参考', flush=True)
    # 实时输入没有未来信息：用当前参考重复填充 12 帧，满足模型未来帧索引。
    # q:(29,)；位置:(30,3)；姿态:(30,4)，四元数排列为 x,y,z,w。
    n = 12
    ev.n_motion_frames = n
    ev.motion_frame_idx = 0
    ev.ref_dof_pos = np.repeat(q[None], n, axis=0).astype(np.float32)
    ev.ref_dof_vel = np.zeros_like(ev.ref_dof_pos)
    ev.ref_global_translation = np.repeat(retarget.initial_pos[None], n, axis=0).astype(np.float32)
    ev.ref_global_rotation_quat_xyzw = np.repeat(retarget.initial_quat[None], n, axis=0).astype(np.float32)
    ev.ref_global_velocity = np.zeros_like(ev.ref_global_translation)
    ev.ref_global_angular_velocity = np.zeros_like(ev.ref_global_translation)
    ev.reset_state_teleport()
    ev._ensure_ref_to_sim_transform_rigid()
    return ev, retarget


# 更新期望姿态，不直接改机器人实际 qpos；参考速度在本原型中设为零。
def set_reference(ev, q, pos, quat):
    ev.ref_dof_pos[:] = q
    ev.ref_dof_vel.fill(0)
    ev.ref_global_translation[:] = pos
    ev.ref_global_rotation_quat_xyzw[:] = quat


# 一个控制周期：观测构造 → ONNX 推理 → 动作缩放/重排 → PD 力矩 → 物理子步。
def policy_step(ev):
    # 官方模型输入 obs、历史 KV 缓存、步号；输出 actions 和新的缓存。
    ev._update_policy()
    ev.counter += 1
    ev._apply_control(sleep=False)
    # 清空官方评估器的离线序列，避免实时运行时内存随时间增长。
    for name, value in vars(ev).items():
        if name.endswith('_seq') and isinstance(value, list):
            value.clear()
    if not np.isfinite(ev.d.qpos).all() or not np.isfinite(ev.actions_onnx).all():
        raise RuntimeError('仿真出现无效数值，已暂停')


def simulation_pd_control(ev, control):
    """默认使用 ONNX 原增益；bridge 仅用于仿真 PD 对比实验。"""
    profile = os.environ.get('HOLOMOTION_SIM_PD', 'model')
    if profile not in ('bridge', 'model'):
        raise ValueError('HOLOMOTION_SIM_PD 只能是 bridge 或 model')
    print(f'[sim] PD={profile}, gravity={ev.m.opt.gravity.tolist()} m/s^2', flush=True)
    if profile == 'model':
        return control

    # 对齐 g1-remote-action/config/bridge.yaml；按关节名映射，不依赖数组顺序。
    leg = {'hip_pitch': (350., 5.), 'hip_roll': (200., 5.),
           'hip_yaw': (200., 5.), 'knee': (300., 10.),
           'ankle_pitch': (300., 5.), 'ankle_roll': (150., 5.)}
    gains = {f'{side}_{joint}_joint': pair
             for side in ('left', 'right') for joint, pair in leg.items()}
    gains.update({f'waist_{axis}_joint': (200., 5.) for axis in ('yaw', 'roll', 'pitch')})
    gains.update({f'{side}_{joint}_joint': (40., 3.)
                  for side in ('left', 'right')
                  for joint in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
                                'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')})
    pairs = np.asarray([gains[name] for name in ev.mjcf_dof_names], dtype=np.float32)

    def apply(sleep=False):
        original_kp, original_kd = ev.kps_mu, ev.kds_mu
        try:
            ev.kps_mu, ev.kds_mu = pairs[:, 0], pairs[:, 1]
            return control(sleep=sleep)
        finally:
            # 真机后端仍读取模型原增益，避免仿真实验改变实机指令。
            ev.kps_mu, ev.kds_mu = original_kp, original_kd

    return apply


def capture_alignment_target(adapter, transport, landmarks):
    """Freeze one human sample; solve reference without inference or robot I/O."""
    frozen = np.asarray(landmarks, dtype=float).copy()
    adapter.calibrate(frozen, strict=False)
    stable = 0
    for _ in range(160):
        before = adapter.q.copy()
        before_pos = adapter.d.xpos[adapter.body_ids].copy()
        q, _, _ = adapter.update(frozen)
        # Near-straight arms have weakly constrained shoulder twist. Freeze a repeatable
        # approximate solution once joint changes <0.002 rad and FK changes <0.5 mm.
        quiet = (np.max(np.abs(q - before)) < .002 and
                 np.max(np.linalg.norm(adapter.d.xpos[adapter.body_ids] - before_pos, axis=1)) < .0005)
        stable = stable + 1 if quiet else 0
        if stable >= 5:
            break
    else:
        raise ValueError('冻结姿态的 IK 未收敛，请调整入镜姿态后重试')
    by_name = dict(zip(adapter.names, q))
    return np.asarray([by_name[n] for n in transport.names], dtype=np.float32)


# 独立约 12 Hz 画面线程；复制状态到自己的 MjData，避免与控制线程共写。
def render_loop(model, pose=None):
    global FRAME
    try:
        d = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.distance = 2.8
        camera.azimuth = 160
        camera.elevation = -12
        fast_preview = BACKEND == 'direct'
        height, width = (240, 424) if fast_preview else (360, 640)
        if fast_preview:
            model.vis.quality.offsamples = 0
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            from OpenGL import GL
            renderer_name = (GL.glGetString(GL.GL_RENDERER) or b'').decode('utf-8', 'replace')
            vendor_name = (GL.glGetString(GL.GL_VENDOR) or b'').decode('utf-8', 'replace')
            lowered = renderer_name.lower()
            if not renderer_name or any(x in lowered for x in ('llvmpipe', 'softpipe', 'swrast', 'software', 'gdi generic')):
                raise RuntimeError(f'禁止软件渲染，当前渲染器：{renderer_name or "未知"}')
            with LOCK:
                STATE.update(render_device=renderer_name, render_vendor=vendor_name,
                             render_hardware=True)
            print(f'[render] hardware renderer: {renderer_name}', flush=True)
            while not STOP.is_set():
                start = time.monotonic()
                with LOCK:
                    source = STATE['display_source']
                    sim = SIM_SNAPSHOT
                    measured = MEASURED_SNAPSHOT
                    measured_stamp = MEASURED_STAMP
                    ref = SNAPSHOT
                if time.monotonic() - measured_stamp > STATE_TIMEOUT_S:
                    measured = None
                # Display selection is independent of robot enable / policy feedback.
                live = pose.snapshot() if (pose is not None and source == 'robot') else None
                if live is not None:
                    measured = (live[0], live[1], ref[2] if ref is not None else None)
                # The bridge copy is invalidated by the control-thread freshness check.
                snap = choose_render_snapshot(source, sim, measured)
                with LOCK:
                    STATE.update(render_source=source, render_available=snap is not None)
                if snap is not None:
                    d.qpos[:] = snap[0]; d.qvel[:] = snap[1]
                    mujoco.mj_forward(model, d)
                    camera.lookat[:] = d.qpos[:3] + [0, 0, .12]
                    renderer.update_scene(d, camera=camera)
                    scene = renderer.scene
                    if fast_preview:
                        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW,
                                     mujoco.mjtRndFlag.mjRND_REFLECTION,
                                     mujoco.mjtRndFlag.mjRND_FOG,
                                     mujoco.mjtRndFlag.mjRND_SKYBOX):
                            scene.flags[flag] = False
                    # 五个绿色点仅用于显示参考位置，不是模型只用了五个点。
                    if snap[2] is not None:
                        for p in snap[2][[0, 6, 12, 22, 29]]:
                            if scene.ngeom >= scene.maxgeom:
                                break
                            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([.025]*3), p, np.eye(3).ravel(), np.array([.1, .9, .6, .8]))
                            scene.ngeom += 1
                    if fast_preview and snap[2] is not None:
                        # Reference-only cyan skeleton; does not change physics.
                        points = np.asarray(snap[2])
                        for a, b in REFERENCE_EDGES:
                            if scene.ngeom >= scene.maxgeom:
                                break
                            if not np.isfinite(points[[a,b]]).all():
                                continue
                            geom = scene.geoms[scene.ngeom]
                            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array([0., .9, 1., 1.]))
                            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                .008, points[a], points[b])
                            scene.ngeom += 1
                    image = renderer.render()
                    if fast_preview:
                        cv2.putText(image, 'CYAN: reference | SOLID: physics', (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 220, 255), 1)
                    ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70 if fast_preview else 80])
                    if ok:
                        FRAME = encoded.tobytes()
                else:
                    # Do not leave a stale robot image or silently show simulation.
                    image = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(image, 'Robot pose unavailable' if source == 'robot' else 'Simulation unavailable',
                                (50, 180), cv2.FONT_HERSHEY_SIMPLEX, .75, (190, 190, 190), 1)
                    ok, encoded = cv2.imencode('.jpg', image)
                    if ok:
                        FRAME = encoded.tobytes()
                render_elapsed = time.monotonic() - start
                with LOCK:
                    STATE['render_ms'] = round(render_elapsed * 1000, 1)
                    STATE['render_size'] = [width, height]
                    STATE['render_fast_preview'] = fast_preview
                    STATE['render_target_hz'] = 30 if BACKEND == 'direct' else 12
                STOP.wait(max(0, 1/(30 if BACKEND == 'direct' else 12) - render_elapsed))
    except Exception:
        with LOCK:
            STATE['error'] = '画面渲染失败：' + traceback.format_exc().splitlines()[-1]
        traceback.print_exc()


# 约 50 Hz 控制线程：先处理复位/校准请求，再判断是否能推进仿真。
def simulation_loop():
    global SNAPSHOT, SIM_SNAPSHOT, MEASURED_SNAPSHOT, MEASURED_STAMP
    transport = None
    alignment = None
    diagnostics = None
    parallel = None
    pose_src = None
    try:
        ev, adapter = make_evaluator()
        print(f"[tracking] enabled={bool(adapter.tracking['enabled'])}, new-frame IK, "
              f"min_cutoff={adapter.tracking['min_cutoff_hz']}Hz beta={adapter.tracking['beta']} "
              f"hold_exit={adapter.tracking['hold_exit_rad']}rad", flush=True)
        # 评测器自带的物理推进方法，留作「纯仿真」模式的实现。
        sim_control = simulation_pd_control(ev, ev._apply_control)
        ev._apply_control = sim_control
        if BACKEND == 'dds':
            from dds_backend import DDSBackend
            transport = DDSBackend(ev)
            ev._apply_control = simulation_pd_control(ev, transport.step)
        elif BACKEND == 'g1ra':
            from g1ra_backend import G1raBackend
            transport = G1raBackend(
                ev, ACTION_ENDPOINT, STATE_ENDPOINT, ACTION_MODE, STATE_MODE,
                lock_lower_body=LOCK_LOWER_BODY, state_timeout_s=STATE_TIMEOUT_S,
                control_mode=G1RA_CONTROL_MODE,
            )
            parallel = ParallelSimulation(ev, sim_control)
            transport.simulation = parallel
            parallel.select_feedback('simulation')
            log_path = os.environ.get('HOLOMOTION_MOTION_LOG',
                str(Path(tempfile.gettempdir()) / 'holomotion-motion-8765.jsonl'))
            diagnostics = MotionDiagnostics(log_path)
            print(f'[diagnostics] 关节记录（1 Hz）: {log_path}', flush=True)
            # 默认不接管：先当纯仿真跑，等前端打开真机开关才下发命令。
            ev._apply_control = sim_control
        with LOCK:
            STATE['backend'] = BACKEND
            STATE['tracking_enabled'] = bool(adapter.tracking['enabled'])
            STATE['g1ra_control_mode'] = G1RA_CONTROL_MODE if BACKEND == 'g1ra' else None
            STATE['feedback_source'] = 'robot' if BACKEND == 'g1ra' else 'simulation'
            STATE['display_source'] = 'robot' if POSE_ENDPOINT or BACKEND == 'g1ra' else 'simulation'
            STATE['robot_view_supported'] = bool(POSE_ENDPOINT or BACKEND == 'g1ra')
            STATE['dds_pairs'] = 0
            # 链路字段先给默认值，前端不必区分「还没轮询」和「未连接」。
            if BACKEND == 'g1ra':
                STATE.update(link_action=False, link_state=False, action_sent=0,
                             state_seq=0, state_valid=False, robot_state=0, state_age_ms=None)
        with LOCK:
            STATE.update(ready=True, stage='模型已就绪，等待摄像头与校准', gpu=ev.policy_session.get_providers()[0],
                         ik_solver='mink/daqp')
            SNAPSHOT = (ev.d.qpos.copy(), ev.d.qvel.copy(), adapter.initial_pos.copy())
            SIM_SNAPSHOT = parallel.snapshot(adapter.initial_pos) if parallel is not None else SNAPSHOT
        # 真机位姿源：给了地址就按机器人实测姿态渲染，否则沿用仿真状态。
        if POSE_ENDPOINT:
            from pose_http import HttpPoseSource
            pose_src = HttpPoseSource(ev.m, POSE_ENDPOINT).start()
            print(f'[pose] 真机位姿源 {POSE_ENDPOINT}', flush=True)
        threading.Thread(target=render_loop, args=(ev.m, pose_src), daemon=True).start()
        last = time.monotonic(); completed = 0; hz_start = last
        last_tick_for_metrics = None
        while not STOP.is_set():
            tick = time.monotonic()
            control_dt = min(.04, max(.001, tick - last))
            last_tick_for_metrics = last
            last = tick
            with LOCK:
                # 复制这一轮状态快照，复位和校准是消费一次的请求标记。
                state = STATE.copy()
                STATE['reset'] = STATE['calibrate'] = STATE['align'] = STATE['cancel_alignment'] = False
            if state['cancel_alignment'] or state['reset']:
                alignment = None
                state['align'] = state['calibrate'] = False
                if BACKEND == 'g1ra':
                    transport.reset()
                    ev._apply_control = sim_control
                    state.update(robot_enabled=False, running=False, calibrated=False)
                    with LOCK:
                        STATE.update(robot_enabled=False, running=False, calibrated=False,
                                     alignment_status='idle', stage='已取消对齐并停止下发')
            if parallel is not None and state['feedback_source'] != transport.feedback_source:
                # HTTP requires pause, cancels alignment and disables publishing.
                # Reset policy/KV observations on simulation data, with no robot I/O.
                transport.feedback_source = state['feedback_source']
                transport.reset()
                parallel.select_feedback('simulation')
                ev.reset_state_teleport()
                ev.obs_builder.reset()
                parallel.seed(ev.d)
                with LOCK:
                    STATE.update(feedback_pending=False, error=None,
                                 stage='策略反馈来源已切换，请重新采集并对齐机器人')
            if parallel is not None:
                parallel.select_feedback(transport.feedback_source if state['robot_enabled'] else 'simulation')
            if state['reset']:
                state['running'] = state['calibrated'] = False
                adapter.q = adapter.nominal.copy()
                adapter.basis = None
                adapter.reset_tracking()
                adapter.fk(adapter.q)
                set_reference(ev, adapter.q, adapter.initial_pos, adapter.initial_quat)
                ev.reset_state_teleport()
                if parallel is not None:
                    parallel.seed(ev.d)
                if transport is not None:
                    transport.reset()
                with LOCK:
                    STATE.update(running=False, calibrated=False, auto_calibrated=False,
                                 fallen=False, error=None, stage='已复位，请校准后开始')
            if state['calibrate']:
                try:
                    if tick - state['last_pose'] > .5:
                        raise ValueError('未收到新鲜人体关键点，请先开启摄像头并完整入镜')
                    # 真机模式走严格门限；纯仿真只求骨架对得上。
                    strict = BACKEND == 'g1ra' or bool(state['robot_enabled'])
                    worst = adapter.calibrate(state['pose'], strict=strict)
                    with LOCK:
                        if strict:
                            # 把试算偏差回报给前端：数值越小说明初始姿态越贴近站姿。
                            STATE.update(calibrated=True, auto_calibrated=False, error=None,
                                         calib_delta_deg=round(float(np.degrees(worst)), 1),
                                         stage=f'校准完成（初始偏差 {np.degrees(worst):.0f}°），可开始跟随')
                        else:
                            STATE.update(calibrated=True, auto_calibrated=False, error=None,
                                         stage='仿真模式：校准完成（未做严格检查）')
                except ValueError as exc:
                    with LOCK:
                        STATE.update(error=str(exc), running=False)
            if state['align'] and BACKEND == 'g1ra':
                # A new click replaces the frozen target; never accumulates angle offsets.
                alignment = None
                state.update(running=False, calibrated=False, robot_enabled=False)
                transport.reset()
                with LOCK:
                    STATE.update(running=False, calibrated=False, robot_enabled=False,
                                 auto_calibrated=False, alignment_status='capturing', error=None,
                                 alignment_error_deg=None, alignment_speed_deg_s=None,
                                 alignment_observed_settled=False)
                try:
                    if tick - state['last_pose'] > .5:
                        raise ValueError('人体关键点过期，请完整入镜并保持姿态后重试')
                    target = capture_alignment_target(adapter, transport, state['pose'])
                    with LOCK:
                        if STATE['cancel_alignment'] or STATE['reset']:
                            raise ValueError('对齐已取消')
                    if not transport.sync_hold():
                        raise ValueError('机器人反馈无效或过期，未开始对齐')
                    alignment = JointAlignment(transport.names, transport._state.q, target, time.monotonic())
                    ev._apply_control = transport.step
                    state.update(robot_enabled=True, running=False, calibrated=False)
                    with LOCK:
                        STATE.update(robot_enabled=True, running=False, calibrated=False,
                                     alignment_status='waiting_policy', error=None)
                except ValueError as exc:
                    ev._apply_control = sim_control
                    with LOCK:
                        STATE.update(alignment_status='failed', stage='未开始实机对齐', error=str(exc))
            # 纯仿真模式不需要人工校准：只要有关键点，就按当前姿态自动建立基准。
            with LOCK:
                need_auto = BACKEND != 'g1ra' and not STATE['robot_enabled'] and not STATE['calibrated']
            if need_auto and state['pose'] is not None and tick - state['last_pose'] <= .5:
                try:
                    adapter.calibrate(state['pose'], strict=False)
                    with LOCK:
                        STATE.update(calibrated=True, auto_calibrated=True, error=None,
                                     stage='仿真模式：已自动匹配骨架，可直接开始跟随')
                except ValueError as exc:
                    with LOCK:
                        STATE.update(error=str(exc))
            # 链路心跳：每轮都刷新（含暂停状态），前端才知道是否连着机器人、有没有在发命令。
            if transport is not None and BACKEND == 'g1ra':
                transport.poll_link()
                with LOCK:
                    MEASURED_SNAPSHOT = transport.measured_snapshot(ev.ref_global_translation[0])
                    MEASURED_STAMP = transport._state_time
                    STATE.update(
                        link_action=bool(transport.action_connected),
                        link_state=bool(transport.state_connected),
                        action_sent=int(transport.sent),
                        state_seq=int(transport.state_seq),
                        state_valid=bool(transport._state_is_valid),
                        simulation_gate=getattr(transport, 'simulation_gate_status', '等待仿真策略验证'),
                        robot_state=int(transport.robot_state),
                        bridge_flags=(0 if transport._state is None else int(transport._state.bridge_flags)),
                        state_age_ms=(None if transport.state_age_ms is None
                                      else round(transport.state_age_ms, 1)),
                    )
            # 真机位姿源健康度：前端据此判断画面到底是真机姿态还是仿真。
            if pose_src is not None:
                with LOCK:
                    STATE['pose_link'] = pose_src.status()
            if alignment is not None:
                if state['running'] and state['calibrated']:
                    # Arrival is diagnostic only; the operator decides when to follow.
                    alignment = None
                    transport.needs_sync = True
                    with LOCK:
                        STATE['alignment_status'] = 'following'
                elif not state['robot_enabled']:
                    alignment = None
                    transport.reset()
                else:
                    try:
                        if not transport.prepare_policy_step(force_measured=True):
                            raise ValueError('实机反馈过期，对齐已停止；请恢复反馈后重新对齐')
                        frame = transport._state
                        command_q = alignment.step(frame.q, frame.dq, transport.state_seq,
                            transport.alignment_control_ready(), time.monotonic())
                        with LOCK:
                            if STATE['cancel_alignment'] or STATE['reset'] or not STATE['robot_enabled']:
                                raise ValueError('对齐已取消')
                            transport.send_alignment(command_q)
                        # Green markers show the frozen target; gray robot remains measured feedback.
                        by_name = dict(zip(transport.names, alignment.target))
                        ref_q = np.asarray([by_name[n] for n in adapter.names])
                        adapter.fk(ref_q)
                        set_reference(ev, ref_q, adapter.d.xpos[adapter.body_ids],
                                      adapter.d.xquat[adapter.body_ids][:, [1, 2, 3, 0]])
                        transport.advance_simulation(command_q)
                        arrived = alignment.status == 'holding'
                        stage = ('已观察到静止，正在保持；可开始跟随' if arrived else
                                 '等待机器人进入 POLICY，当前仅发送起始保持角' if alignment.status == 'waiting_policy' else
                                 '正在对齐（误差仅记录，可直接开始跟随）')
                        diagnostics.record(transport, 'alignment', reference=alignment.target, alignment=alignment,
                                           ik=adapter.ik.last_diagnostics)
                        with LOCK:
                            STATE.update(calibrated=alignment.status != 'waiting_policy', alignment_status=alignment.status,
                                alignment_observed_settled=arrived,
                                alignment_error_deg=round(float(np.degrees(alignment.error)), 2),
                                alignment_speed_deg_s=round(float(np.degrees(alignment.speed)), 2),
                                stage=stage, error=None)
                    except ValueError as exc:
                        alignment = None
                        transport.reset()
                        ev._apply_control = sim_control
                        with LOCK:
                            STATE.update(robot_enabled=False, running=False, calibrated=False,
                                         alignment_status='failed', stage='对齐已停止', error=str(exc))
                    with LOCK:
                        SNAPSHOT = (ev.d.qpos.copy(), ev.d.qvel.copy(), ev.ref_global_translation[0].copy())
                        SIM_SNAPSHOT = parallel.snapshot(ev.ref_global_translation[0])
                        STATE['local_simulation'] = parallel.status()
                    STOP.wait(max(0, .02 - (time.monotonic() - tick)))
                    continue
            # 真机开关：关闭时用评测器自带物理推进（纯仿真，不下发任何命令），
            # 打开时才把 _apply_control 换成 transport.step 去发动作。
            if BACKEND == 'g1ra' and transport is not None:
                want_robot = bool(state.get('robot_enabled'))
                target_control = transport.step if want_robot else sim_control
                if ev._apply_control != target_control:
                    if want_robot:
                        if transport.sync_hold():
                            ev._apply_control = target_control
                            with LOCK:
                                STATE['stage'] = '已从机器人当前姿态建立起点，可开始跟随'
                        else:
                            state.update(robot_enabled=False, running=False)
                            with LOCK:
                                STATE.update(robot_enabled=False, running=False,
                                             error='机器人状态无效或过期，未接管；请恢复状态反馈后重试')
                    else:
                        ev._apply_control = target_control
                        transport.reset()
                        with LOCK:
                            STATE['stage'] = '已切回纯仿真，不再下发命令'
            # 开始、已校准、未跌倒三个条件同时满足才跟随。
            active = state['running'] and state['calibrated'] and not state['fallen']
            # 超过 0.5 秒没收到关键点则冻结物理推进；采用接收时间，不用浏览器时钟。
            if active and tick - state['last_pose'] <= .5:
                try:
                    real_control = BACKEND == 'g1ra' and state['robot_enabled']
                    if real_control:
                        if transport.needs_sync and not transport.sync_hold():
                            raise ValueError('无法获取新鲜实机姿态，已暂停；请恢复反馈后重新开始')
                        if not transport.prepare_policy_step():
                            raise ValueError('实机状态过期，已暂停；请恢复反馈后重新开始')
                    # 人体关键点 → IK 参考关节角及其正向运动学位置/旋转。
                    sim_time_before = float(ev.d.time)
                    mapping_start = time.monotonic()
                    q, pos, quat = adapter.update(state['pose'], dt=control_dt,
                        sample_id=state['sequence'], timestamp=state['pose_time'])
                    if real_control:
                        q = transport.reference_target(q, .02 * adapter.map['max_rate_rad_s'])
                        adapter.fk(q)
                        pos = adapter.d.xpos[adapter.body_ids].copy()
                        quat = adapter.d.xquat[adapter.body_ids][:, [1, 2, 3, 0]].copy()
                    mapping_ms = (time.monotonic() - mapping_start) * 1000
                    set_reference(ev, q, pos, quat)
                    if real_control and transport._align_reference:
                        ev._ref_to_sim_ready = False
                        ev._ensure_ref_to_sim_transform_rigid()
                        transport._align_reference = False
                    start = time.monotonic()
                    policy_step(ev)
                    if real_control:
                        by_name = dict(zip(adapter.names, q))
                        diagnostics.record(transport, 'following',
                            reference=[by_name[n] for n in transport.names],
                            ik=adapter.ik.last_diagnostics)
                    if transport is not None:
                        with LOCK:
                            STATE['dds_pairs'] = transport.pairs
                            if BACKEND == 'g1ra':
                                # 真机模式额外暴露桥回传的状态年龄与状态机状态码
                                STATE['state_age_ms'] = (
                                    None if transport.state_age_ms is None
                                    else round(transport.state_age_ms, 1)
                                )
                                STATE['robot_state'] = transport.robot_state
                    elapsed = (time.monotonic() - start) * 1000
                    # 用骨盆高度做简单跌倒判断；不是完整的稳定性评估指标。
                    # 真机模式下没有可信的仿真基座高度，跌倒判断交给操作员与急停。
                    sim_feedback = not real_control or transport.feedback_source == 'simulation'
                    fallen = bool(ev.d.qpos[2] < .45) if (BACKEND != 'g1ra' or sim_feedback) else False
                    completed += 1
                    with LOCK:
                        # inference_ms 实际统计推理加物理推进耗时，不是纯神经网络耗时。
                        STATE.update(steps=STATE['steps']+1, inference_ms=round(elapsed, 1),
                            mapping_ms=round(mapping_ms, 1),
                            simulated_step_ms=round((float(ev.d.time)-sim_time_before)*1000, 2),
                            control_interval_ms=round((tick-last_tick_for_metrics)*1000, 2) if last_tick_for_metrics is not None else None,
                            pose_filter_held=bool(adapter.pose_filter.held),
                            pose_filter_rejected=int(adapter.pose_filter.rejected),
                            reference_tracking_error_deg=round(float(np.max(np.abs(q-ev.d.qpos[adapter.qadr])))*180/np.pi,1),
                            reference_max_rate_rad_s=adapter.map['max_rate_rad_s'],
                            control_work_ms=round((time.monotonic()-tick)*1000, 1),
                            pelvis_height=round(float(ev.d.qpos[2]), 3),
                            reference_delta=round(float(np.linalg.norm(q-adapter.nominal)), 3),
                            left_shoulder_reference=round(float(q[adapter.names.index('left_shoulder_roll_joint')]), 3),
                            left_shoulder_actual=round(float(ev.d.qpos[ev.m.joint('left_shoulder_roll_joint').qposadr[0]]), 3),
                            stage='正在跟随' if not fallen else '机器人跌倒，已暂停，请复位',
                            fallen=bool(fallen), running=not fallen, error=None,
                            ik=adapter.ik.last_diagnostics)
                except ValueError as exc:
                    with LOCK:
                        STATE.update(stage='关键点不完整，仿真暂停', error=str(exc))
                        if BACKEND == 'g1ra' and state['robot_enabled']:
                            STATE.update(running=False, stage='实机跟随已暂停')
                            transport.needs_sync = True
            elif active:
                with LOCK:
                    STATE['stage'] = '摄像头数据过期，仿真暂停'
            if BACKEND == 'g1ra' and state['robot_enabled'] and (
                not active or tick - state['last_pose'] > .5
            ):
                transport.needs_sync = True
            with LOCK:
                SNAPSHOT = (ev.d.qpos.copy(), ev.d.qvel.copy(), ev.ref_global_translation[0].copy())
                SIM_SNAPSHOT = parallel.snapshot(ev.ref_global_translation[0]) if parallel is not None else SNAPSHOT
                if parallel is not None:
                    STATE['local_simulation'] = parallel.status()
                if tick-hz_start > 1:
                    STATE['policy_hz'] = round(completed/(tick-hz_start), 1)
                    completed = 0; hz_start = tick
            STOP.wait(max(0, .02-(time.monotonic()-tick)))
    except Exception:
        traceback.print_exc()
        with LOCK:
            STATE.update(error=traceback.format_exc().splitlines()[-1], running=False, stage='控制服务出错，已停止')
    finally:
        if pose_src is not None:
            pose_src.stop()
        if diagnostics is not None:
            diagnostics.close()
        if transport is not None:
            transport.close()


# 浏览器通过 HTTP 与此处理器通信；它不直接执行模型推理。
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def log_message(self, *_):
        pass

    def end_headers(self):
        if not self.path.startswith(('/api/', '/frame.jpg')):
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def respond(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(data)))
        self.end_headers(); self.wfile.write(data)

    # GET /api/status: JSON 状态；GET /frame.jpg: JPEG；其他路径: web 静态资源。
    def do_GET(self):
        if self.path == '/api/status':
            with LOCK:
                state = {k:v for k,v in STATE.items() if k not in ('pose',)}
            state['pose_age_ms'] = round((time.monotonic()-state['last_pose'])*1000) if state['last_pose'] else None
            self.respond(state)
        elif self.path.startswith('/frame.jpg'):
            data = FRAME
            if not data:
                self.send_error(503); return
            self.send_response(200); self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)
        else:
            super().do_GET()

    # POST /api/pose: 人体关键点；POST /api/control: 按钮控制请求。
    def do_POST(self):
        # Same-origin only; this local service is never exposed on the LAN.
        origin = self.headers.get('Origin')
        port=self.server.server_port
        if origin and origin not in (f'http://localhost:{port}', f'http://127.0.0.1:{port}'):
            self.respond({'error':'origin rejected'},403); return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size < 20000:
                raise ValueError('invalid payload size')
            payload = json.loads(self.rfile.read(size))
            with LOCK:
                if self.path == '/api/pose':
                    points = payload['landmarks']
                    array = np.asarray(points,dtype=float)
                    # 协议要求恰好 33 行，每行 [x,y,z,visibility]，且所有数值有限。
                    if array.shape != (33,4) or not np.isfinite(array).all():
                        raise ValueError('invalid landmarks')
                    seq = int(payload['sequence'])
                    # 帧序号只允许递增，防止旧帧覆盖新帧；新摄像头会将序号重置。
                    if seq <= STATE['sequence']:
                        raise ValueError('out of order frame')
                    received = time.monotonic()
                    pose_time = float(payload.get('timestamp_ms', received * 1000.)) / 1000.
                    if not np.isfinite(pose_time):
                        raise ValueError('invalid pose timestamp')
                    source_clock = 'timestamp_ms' in payload
                    if source_clock and STATE.get('pose_source_clock') and STATE['sequence'] >= 0 and pose_time <= STATE['pose_time']:
                        raise ValueError('out of order pose timestamp')
                    low = [label for i, label in LANDMARK_LABELS.items()
                           if array[i,3] < VISIBILITY_FLOOR]
                    POSE_QUALITY_WINDOW.append((received, bool(low)))
                    while POSE_QUALITY_WINDOW and received-POSE_QUALITY_WINDOW[0][0] > 1.:
                        POSE_QUALITY_WINDOW.popleft()
                    STATE.update(pose_low_visibility=low,
                        pose_min_visibility=round(float(min(array[i,3] for i in LANDMARK_LABELS)),3),
                        pose_low_visibility_percent=round(100*sum(x[1] for x in POSE_QUALITY_WINDOW)/len(POSE_QUALITY_WINDOW),1),
                        pose_input_window_frames=len(POSE_QUALITY_WINDOW))
                    STATE.update(pose=points,last_pose=received,pose_time=pose_time,
                                 pose_source_clock=source_clock,sequence=seq)
                elif self.path == '/api/control':
                    action=payload.get('action')
                    if action == 'start':
                        if STATE.get('feedback_pending'):
                            raise ValueError('策略反馈正在切换，请稍后重新对齐')
                        if not STATE['ready'] or not STATE['calibrated'] or STATE['fallen']:
                            raise ValueError('请等待模型就绪并先采集姿态，跌倒时先复位')
                        if BACKEND == 'g1ra' and (
                            not STATE['robot_enabled'] or STATE['alignment_status'] not in ('moving', 'holding', 'following')
                        ):
                            raise ValueError('请先采集姿态并启动机器人对齐')
                        STATE['running']=True
                    elif action == 'pause':
                        STATE.update(running=False,stage='已暂停')
                        if BACKEND == 'g1ra' and (STATE['align'] or STATE['alignment_status'] in ('capturing', 'waiting_policy', 'moving', 'holding')):
                            STATE.update(cancel_alignment=True, align=False, calibrated=False, robot_enabled=False)
                    # HTTP 线程只设置请求标记，真正复位/校准由仿真线程执行。
                    elif action in ('reset','calibrate'):
                        if action == 'calibrate' and BACKEND == 'g1ra':
                            raise ValueError('实机校准流程已更新，请刷新页面后点击「采集并对齐机器人（会运动）」')
                        STATE[action]=True
                    elif action == 'align':
                        if STATE.get('feedback_pending'):
                            raise ValueError('策略反馈正在切换，请稍后重新对齐')
                        if BACKEND != 'g1ra' or not STATE['ready']:
                            raise ValueError('实机对齐仅在机器人连接服务就绪后可用')
                        if payload.get('move_robot') is not True:
                            raise ValueError('请通过标明「会运动」的对齐按钮启动')
                        STATE.update(align=True, running=False, calibrated=False, cancel_alignment=False)
                    elif action == 'set_display_source':
                        source = payload.get('source')
                        if source not in ('robot', 'simulation'):
                            raise ValueError('未知画面来源')
                        if source == 'robot' and not STATE.get('robot_view_supported'):
                            raise ValueError('当前服务未配置实机位姿来源')
                        STATE['display_source'] = source
                    elif action == 'set_feedback_source':
                        source = payload.get('source')
                        if BACKEND != 'g1ra' or not STATE['ready']:
                            raise ValueError('策略反馈切换仅在实机服务就绪后可用')
                        if source not in ('robot', 'simulation'):
                            raise ValueError('未知策略反馈来源')
                        if STATE['running'] or STATE.get('align') or STATE['alignment_status'] in (
                            'capturing', 'waiting_policy', 'moving', 'holding'
                        ):
                            raise ValueError('请先暂停跟随或停止对齐，再切换策略反馈')
                        if source != STATE.get('feedback_source', 'robot'):
                            STATE.update(feedback_source=source, feedback_pending=True,
                                         cancel_alignment=True, robot_enabled=False, running=False,
                                         calibrated=False, auto_calibrated=False, alignment_status='idle')
                    elif action == 'set_robot':
                        enabled = bool(payload.get('enabled'))
                        if enabled:
                            # 没校准就接管，等于把未验证的初始姿态直接发给机器人。
                            if not (STATE['ready'] and STATE['calibrated']):
                                raise ValueError('请先完成校准，再打开真机开关')
                            # 仿真模式自动建的基准没过严格门限，切真机前必须重来一次。
                            if STATE['auto_calibrated']:
                                raise ValueError('请先采集人体姿态并完成机器人对齐')
                        STATE['robot_enabled'] = enabled
                        if not enabled:
                            STATE.update(cancel_alignment=True, align=False, calibrated=False, running=False)
                    elif action == 'new_camera':
                        STATE.update(sequence=-1,running=False,calibrated=False,auto_calibrated=False,
                                     cancel_alignment=True, align=False, robot_enabled=False,
                                     pose=None, last_pose=0., pose_time=0., pose_source_clock=False)
                    else:
                        raise ValueError('unknown action')
                else:
                    raise ValueError('unknown endpoint')
            self.respond({'ok':True})
        except (ValueError,KeyError,TypeError) as exc:
            self.respond({'error':str(exc)},400)


# 程序入口：后台线程加载模型，主线程立即监听本机 8765 端口。
# 页面能访问与模型已加载是两个状态，应查看 /api/status 的 ready。
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='上半身模式：摄像头 → IK → HoloMotion motion 策略 → MuJoCo 或真机（不加载全身 velocity 策略）',
        epilog='g1ra 后端把动作发给机器人侧的 g1-remote-action 桥；安全限位/断流保护都在桥里。',
    )
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--backend', choices=('direct','dds','g1ra'), default='direct')
    parser.add_argument('--action-endpoint', default=ACTION_ENDPOINT,
                        help='g1ra：动作帧目标地址（桥 bind 时这里用 connect）')
    parser.add_argument('--state-endpoint', default=STATE_ENDPOINT,
                        help='g1ra：状态帧来源地址')
    parser.add_argument('--action-mode', choices=('connect','bind'), default=ACTION_MODE)
    parser.add_argument('--state-mode', choices=('connect','bind'), default=STATE_MODE)
    parser.add_argument('--no-lower-body-lock', dest='lock_lower_body', action='store_false',
                        help='g1ra：不把下肢锁在站姿（默认锁定，只跟随上肢与躯干）')
    parser.add_argument('--state-timeout', type=float, default=STATE_TIMEOUT_S,
                        help='g1ra：超过该秒数没有机器人状态就停止下发动作')
    parser.add_argument('--g1ra-control-mode', choices=('arm_sdk', 'full_body'), default=G1RA_CONTROL_MODE,
                        help='g1ra：机器人桥接口；arm_sdk 仅手臂，full_body 为全身低层状态机')
    parser.add_argument('--retarget-config', default=RETARGET_CONFIG_PATH,
                        help='校准门限与映射限幅 YAML（默认 webcam_teleop/retarget_config.yaml）')
    parser.add_argument('--pose-endpoint', default=POSE_ENDPOINT,
                        help='真机位姿源（camview joint_relay 的 HTTP 地址，如 http://192.168.110.165:48790）；'
                             '给了就用机器人实测姿态渲染 MuJoCo，留空则用仿真状态')
    args=parser.parse_args()
    BACKEND = args.backend
    ACTION_ENDPOINT = args.action_endpoint
    STATE_ENDPOINT = args.state_endpoint
    ACTION_MODE = args.action_mode
    STATE_MODE = args.state_mode
    LOCK_LOWER_BODY = args.lock_lower_body
    STATE_TIMEOUT_S = args.state_timeout
    G1RA_CONTROL_MODE = args.g1ra_control_mode
    POSE_ENDPOINT = args.pose_endpoint
    RETARGET_CONFIG_PATH = args.retarget_config
    if args.backend == 'direct' and args.port == 8766 and not args.retarget_config:
        RETARGET_CONFIG_PATH = str(Path(__file__).with_name('retarget_low_latency.yaml'))
    simulation_thread = threading.Thread(target=simulation_loop,daemon=True)
    simulation_thread.start()
    print(f'Open http://localhost:{args.port}',flush=True)
    try:
        ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
    except KeyboardInterrupt:
        STOP.set()
        simulation_thread.join(timeout=10)