"""Offline reference / optional ONNX+MuJoCo validation. Never opens robot transport."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys
import time

import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'webcam_teleop'), str(ROOT/'webcam_teleop/tests')]
from retarget import WebcamRetarget
from arm_pose_fixtures import BASE, POSES, blend, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', action='store_true')
    args = parser.parse_args()
    report = dict(scope='offline only; no hardware actuation',
                  versions={n: importlib.metadata.version(n) for n in ('mink', 'mujoco', 'daqp', 'qpsolvers')}, cases={})
    if args.policy:
        import server
        ev, adapter = server.make_evaluator()
        ev._apply_control = server.simulation_pd_control(ev, ev._apply_control)
        factory = lambda: adapter
    else:
        m = mujoco.MjModel.from_xml_path(str(ROOT/'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
        r = yaml.safe_load((ROOT/'checkpoints/HoloMotion_motion_tracking_model_v2.0.0/config.yaml').read_text())['robot']
        factory = lambda: WebcamRetarget(m, r['dof_names'], r['body_names'],
                                        [r['init_state']['default_joint_angles'][n] for n in r['dof_names']])
    for name, goal in POSES.items():
        a = factory()
        a.calibrate(BASE, strict=False)
        if args.policy:
            server.set_reference(ev, a.q, a.initial_pos, a.initial_quat)
            ev.reset_state_teleport(); ev.obs_builder.reset(); ev._ref_to_sim_ready = False
            ev._ensure_ref_to_sim_transform_rigid()
        times = []; min_height = 10.; max_step = 0.; trace = []
        for i in range(300):
            sample = i*2//5; stamp = sample*.05
            p = blend(BASE, goal, float(np.clip((stamp-1)/2, 0, 1)))
            before = a.q.copy(); start = time.perf_counter()
            q, pos, quat = a.update(p, dt=.02, sample_id=sample, timestamp=stamp)
            if i == 0 or sample != (i-1)*2//5:
                times.append((time.perf_counter()-start)*1000)
            max_step = max(max_step, float(np.abs(q-before).max()))
            if args.policy:
                server.set_reference(ev, q, pos, quat); server.policy_step(ev)
                mujoco.mj_forward(ev.m, ev.d)
                min_height = min(min_height, float(ev.d.qpos[2]))
            trace.append(dict(q=q.tolist(), reference=errors(a, p)))
            if args.policy and min_height < .45:
                break
        case = dict(reference=errors(a, goal),
                    solve_ms_p50=float(np.percentile(times, 50)),
                    solve_ms_p95=float(np.percentile(times, 95)),
                    max_reference_step_rad=max_step, steps=len(trace),
                    final_q=a.q.tolist(), solver=a.ik.last_diagnostics)
        if args.policy:
            case.update(actual=errors(a, goal, ev.d), min_pelvis_height_m=min_height,
                        fallen=min_height < .45)
        report['cases'][name] = case
        print(name, json.dumps(case), flush=True)
        (HERE/('policy-results.json' if args.policy else 'reference-results.json')).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
