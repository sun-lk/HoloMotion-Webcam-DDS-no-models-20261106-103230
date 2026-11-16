"""Deterministic old/new reference comparison; no policy, sockets or robot I/O."""
import json
import subprocess
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'webcam_teleop'))
from retarget import WebcamRetarget
from test_tracking_filter import landmarks


def pose(amount):
    p = landmarks()
    p[13, :3] = [.22 + .24 * amount, -.23 - .27 * amount, -.03 * amount]
    p[15, :3] = [.22 + .48 * amount, .02 - .52 * amount, -.04 * amount]
    return p


def run():
    source = subprocess.check_output(['git', 'show', '3cf849f:webcam_teleop/retarget.py'], cwd=ROOT).decode()
    baseline = types.ModuleType('baseline_retarget')
    exec(compile(source, 'baseline_retarget.py', 'exec'), baseline.__dict__)
    robot = yaml.safe_load((ROOT / 'checkpoints/HoloMotion_motion_tracking_model_v2.0.0/config.yaml').read_text())['robot']
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
    names = robot['dof_names']
    defaults = [robot['init_state']['default_joint_angles'][n] for n in names]
    arms = [i for i, n in enumerate(names) if 'shoulder_' in n or 'elbow_' in n]
    wrists = [robot['body_names'].index(s + '_wrist_yaw_link') for s in ('left', 'right')]
    report = dict(baseline_commit='3cf849f', scope='synthetic reference/IK only; no policy or real robot',
        camera_hz=20, control_hz=50, static_noise_std_m=.003, seed=20261105, cases={})
    for case in ('static_straight', 'static_raised', 'slow_sweep'):
        old = baseline.WebcamRetarget(model, names, robot['body_names'], defaults)
        new = WebcamRetarget(model, names, robot['body_names'], defaults)
        initial = .65 if case == 'static_raised' else 0.
        old.calibrate(pose(initial), strict=False)
        new.calibrate(pose(initial), strict=False)
        rng = np.random.default_rng(20261105)
        samples = [[], []]; positions = [[], []]
        previous_frame = -1
        for tick in range(500):
            t = tick / 50.
            frame = int(t * 20 + 1e-8)
            amount = min(.8, max(0., (frame / 20. - 2.) * .8 / 3.)) if case == 'slow_sweep' else initial
            if frame != previous_frame:
                p = pose(amount)
                if case != 'slow_sweep' and t >= 2.:
                    p[PoseIndices, :3] += rng.normal(0., .003, (len(PoseIndices), 3))
                previous_frame = frame
            q0, p0, _ = old.update(p)
            q1, p1, _ = new.update(p, sample_id=frame, timestamp=frame / 20.)
            for k, (q, xyz) in enumerate(((q0, p0), (q1, p1))):
                samples[k].append(q[arms].copy())
                positions[k].append(xyz[wrists].copy())
        samples, positions = np.asarray(samples), np.asarray(positions)
        result = {}
        if case != 'slow_sweep':
            window = samples[:, 200:]
            rate_rms = np.sqrt(np.mean((np.diff(window, axis=1) * 50.) ** 2, axis=(1, 2)))
            result.update(baseline_joint_rate_rms_deg_s=float(np.degrees(rate_rms[0])),
                filtered_joint_rate_rms_deg_s=float(np.degrees(rate_rms[1])),
                rate_rms_reduction_percent=float(100. * (1. - rate_rms[1] / rate_rms[0])),
                baseline_max_joint_std_deg=float(np.max(np.degrees(np.std(window[0], axis=0)))),
                filtered_max_joint_std_deg=float(np.max(np.degrees(np.std(window[1], axis=0)))))
        else:
            amplitude = np.abs(np.mean(samples[0, 300:350], axis=0) - np.mean(samples[0, 50:90], axis=0))
            idx = int(np.argmax(amplitude))
            crossing = []
            for k in (0, 1):
                low = np.mean(samples[k, 50:90, idx]); high = np.mean(samples[k, 300:350, idx])
                curve = (samples[k, :, idx] - low) / (high - low)
                crossing.append(float(np.where(curve[100:] >= .5)[0][0] / 50. + 2.))
            result.update(measured_joint=names[arms[idx]], baseline_half_sweep_s=crossing[0],
                filtered_half_sweep_s=crossing[1], additional_half_sweep_delay_ms=1000.*(crossing[1]-crossing[0]))
        result['mean_wrist_reference_difference_mm'] = (
            np.mean(np.linalg.norm(positions[1, 300:] - positions[0, 300:], axis=2), axis=0) * 1000.).tolist()
        report['cases'][case] = result
        print(case, json.dumps(result), flush=True)
    output = ROOT / 'diagnostics/jitter-filter-2026-11-05.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


PoseIndices = [11, 12, 13, 14, 15, 16, 23, 24]
if __name__ == '__main__':
    run()
