"""Offline mapping audit: ideal landmarks and model-generated reachable poses.

No HTTP, transport, policy inference, robot writes, or production config changes.
"""
from pathlib import Path
import copy
import json
import re
import sys

import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'webcam_teleop'))
from retarget import WebcamRetarget, unit
import retarget_config

M = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
ROBOT = yaml.safe_load((ROOT / 'checkpoints/HoloMotion_motion_tracking_model_v2.0.0/config.yaml').read_text())['robot']
NAMES = ROBOT['dof_names']
DEFAULT = np.array([ROBOT['init_state']['default_joint_angles'][n] for n in NAMES])
origin = 'checkpoint YAML'
log = Path('/tmp/holomotion-sim-8766-20261105-174955.log')
if log.exists():
    text = log.read_text()
    values = re.search(r'default_joint_pos:\s*\[([^]]+)\]', text)
    names = re.search(r'joint_names: ([^\n]+)', text)
    if values and names:
        defaults = dict(zip(names[1].strip().split(', '), np.fromstring(values[1], sep=' ')))
        DEFAULT = np.array([defaults[n] for n in NAMES])
        origin = 'running simulation ONNX metadata in startup log'


def pose(left_upper, left_fore, right_upper=None, right_fore=None):
    p = np.zeros((33, 4)); p[:, 3] = .99
    p[11, :3], p[12, :3] = [.2, -.5, 0], [-.2, -.5, 0]
    p[23, :3], p[24, :3] = [.13, 0, 0], [-.13, 0, 0]
    if right_upper is None:
        right_upper = np.asarray(left_upper) * [1, -1, 1]
        right_fore = np.asarray(left_fore) * [1, -1, 1]
    for (s,e,w), (u,v) in zip(((11,13,15),(12,14,16)),
                             ((left_upper,left_fore),(right_upper,right_fore))):
        # Calibration basis maps human [x,y,z] -> robot [-z,x,-y].
        def human(d):
            x,y,z = unit(d)
            return np.array([y,-z,-x])
        p[e,:3] = p[s,:3] + .27*human(u)
        p[w,:3] = p[e,:3] + .25*human(v)
    return p


BASE = pose([0,0,-1], [0,0,-1])


def blend_pose(start, end, fraction):
    """Rotate bone directions along an arc; never shrink bones through zero."""
    p = start.copy()
    for s,e,w in ((11,13,15),(12,14,16)):
        for parent,child,length in ((s,e,.27),(e,w,.25)):
            u,v = unit(start[child,:3]-start[parent,:3]), unit(end[child,:3]-end[parent,:3])
            dot=float(np.clip(u@v,-1,1))
            if dot < -.99999:
                axis=np.array([0.,0.,1.])
                if abs(axis@u)>.9:axis=np.array([1.,0.,0.])
                axis=unit(axis-(axis@u)*u)
                direction=np.cos(np.pi*fraction)*u+np.sin(np.pi*fraction)*axis
            elif dot > .99999:
                direction=unit((1-fraction)*u+fraction*v)
            else:
                angle=np.arccos(dot)
                direction=(np.sin((1-fraction)*angle)*u+np.sin(fraction*angle)*v)/np.sin(angle)
            p[child,:3]=p[parent,:3]+length*direction
    return p


POSES = dict(arms_down_straight=BASE,
             side_raise_straight=pose([0,1,0],[0,1,0]),
             forward_raise_straight=pose([1,0,0],[1,0,0]),
             overhead_straight=pose([0,0,1],[0,0,1]),
             hello_forearms_up=pose([0,1,0],[0,0,1]))


def adapter(cfg=None):
    return WebcamRetarget(M, NAMES, ROBOT['body_names'], DEFAULT, config=copy.deepcopy(cfg or retarget_config.load()))


def metrics(a, p):
    a.fk(a.q)
    result = {}
    for side,(s,e,w) in [('left',(11,13,15)),('right',(12,14,16))]:
        arm = a.arms[side]
        u = unit(p[e,:3]-p[s,:3]) @ a.basis.T
        v = unit(p[w,:3]-p[e,:3]) @ a.basis.T
        su = unit(a.d.xpos[arm['elbow']]-a.d.xpos[arm['shoulder']])
        sv = unit(a.d.xpos[arm['wrist']]-a.d.xpos[arm['elbow']])
        err = lambda x,y: float(np.degrees(np.arccos(np.clip(x@y,-1,1))))
        q = a.q[arm['indices']]
        model_bounds = M.jnt_range[[M.joint(NAMES[i]).id for i in arm['indices']]]
        result[side] = dict(upper_direction_error_deg=err(u,su),
                            fore_direction_error_deg=err(v,sv),
                            requested_bend_deg=err(u,v), actual_bend_deg=err(su,sv),
                            joint_deg=np.degrees(q).tolist(),
                            distance_to_real_joint_limit_deg=np.degrees(np.minimum(q-model_bounds[:,0],model_bounds[:,1]-q)).tolist())
    return result


cfg = retarget_config.load()
variants = {'current': cfg}
for name, section, key, value in (
    ('no_input_filter','tracking','enabled',0),
    ('no_joint_margin','mapping','joint_margin_rad',0),
    ('no_continuity_penalty','mapping','ik_continuity_weight',0),
    ('lower_damping','mapping','ik_damping',.0001),
    ('more_iterations','mapping','ik_iterations',30),
):
    c=copy.deepcopy(cfg);c[section][key]=value;variants[name]=c
variants['wider_bounds_offline_only']=copy.deepcopy(cfg)

report = dict(default_source=origin, scope='ideal input geometric/IK audit only; no learned policy or hardware proof',
              joint_limits_deg={n:np.degrees(M.jnt_range[M.joint(n).id]).tolist()
                                for n in NAMES if any(k in n for k in ('shoulder','elbow','waist'))},
              results={})
for name,p in POSES.items():
    case = {}
    for v,c in variants.items():
        a=adapter(c);a.calibrate(BASE, strict=False)
        if v=='wider_bounds_offline_only':
            for arm in a.arms.values():arm['bounds'][:]=[-np.pi,np.pi]
        # 20 Hz ideal camera; ramp over 2 s, hold 4 s, 50 Hz reference loop.
        for i in range(325):
            sample=i*2//5;stamp=sample*.05
            blend=np.clip((stamp-.5)/2,0,1)
            a.update(blend_pose(BASE,p,blend),dt=.02,sample_id=sample,timestamp=stamp)
        case[v]=metrics(a,p)
        case[v]['q_reference']=a.q.tolist()
    report['results'][name]=case
    (HERE/'ik-results.json').write_text(json.dumps(report,indent=2))
    print(name, json.dumps({k:{'upper_deg':round(v['left']['upper_direction_error_deg'],2),
                             'fore_deg':round(v['left']['fore_direction_error_deg'],2),
                             'bend_deg':round(v['left']['actual_bend_deg'],2)} for k,v in case.items()}),flush=True)

# Generate a guaranteed reachable arm pose from the same model and audit remapping.
generator=adapter()
q=DEFAULT.copy()
for side,sign in [('left',1),('right',-1)]:
    for joint,value in [('shoulder_pitch',0),('shoulder_roll',sign*np.pi/2),('shoulder_yaw',0),('elbow',np.pi/2)]:
        q[NAMES.index(f'{side}_{joint}_joint')]=value
generator.fk(q)
dirs=[]
for side in ('left','right'):
    arm=generator.arms[side]
    dirs.extend([unit(generator.d.xpos[arm['elbow']]-generator.d.xpos[arm['shoulder']]),
                 unit(generator.d.xpos[arm['wrist']]-generator.d.xpos[arm['elbow']])])
p=pose(*dirs)
a=adapter();a.calibrate(BASE,strict=False)
for i in range(300):a.update(blend_pose(BASE,p,min(1,i/100)),dt=.02,sample_id=i,timestamp=i*.02)
report['known_reachable_model_pose']=dict(requested_joints_deg={n:float(np.degrees(q[i])) for i,n in enumerate(NAMES) if any(k in n for k in ('shoulder','elbow'))},result=metrics(a,p))
print('KNOWN_REACHABLE',json.dumps(report['known_reachable_model_pose']),flush=True)
(HERE/'ik-results.json').write_text(json.dumps(report,indent=2))
