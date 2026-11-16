"""Real ONNX + local physics for ideal side raise / straight arm / hello poses.

No sockets, HTTP, DDS, publishers, or real robot interaction.
"""
from pathlib import Path
import json
import sys
import time

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'webcam_teleop'))
import server


def unit(v):
    v=np.asarray(v,dtype=float)
    return v/np.linalg.norm(v)


def pose(upper,fore):
    p=np.zeros((33,4));p[:,3]=.99
    p[11,:3],p[12,:3]=[.2,-.5,0],[-.2,-.5,0]
    p[23,:3],p[24,:3]=[.13,0,0],[-.13,0,0]
    for (s,e,w),sign in [((11,13,15),1),((12,14,16),-1)]:
        def human(d):
            x,y,z=unit(d)*[1,sign,1]
            return np.array([y,-z,-x])
        p[e,:3]=p[s,:3]+.27*human(upper)
        p[w,:3]=p[e,:3]+.25*human(fore)
    return p


BASE=pose([0,0,-1],[0,0,-1])


def blend(start,end,fraction):
    p=start.copy()
    for s,e,w in ((11,13,15),(12,14,16)):
        for parent,child,length in ((s,e,.27),(e,w,.25)):
            u,v=unit(start[child,:3]-start[parent,:3]),unit(end[child,:3]-end[parent,:3])
            dot=float(np.clip(u@v,-1,1))
            if dot>.99999:direction=unit((1-fraction)*u+fraction*v)
            elif dot<-.99999:
                axis=np.array([0.,0.,1.]);axis=unit(axis-(axis@u)*u)
                direction=np.cos(np.pi*fraction)*u+np.sin(np.pi*fraction)*axis
            else:
                angle=np.arccos(dot)
                direction=(np.sin((1-fraction)*angle)*u+np.sin(fraction*angle)*v)/np.sin(angle)
            p[child,:3]=p[parent,:3]+length*direction
    return p


ev,a=server.make_evaluator()
ev._apply_control=server.simulation_pd_control(ev,ev._apply_control)
report=dict(scope='ideal input through real ONNX and local MuJoCo, no hardware validation',cases={})
for name,target in [('side_raise_straight',pose([0,1,0],[0,1,0])),
                    ('forward_raise_straight',pose([1,0,0],[1,0,0])),
                    ('hello_forearms_up',pose([0,1,0],[0,0,1]))]:
    a.q=a.nominal.copy();a.fk(a.q);a.calibrate(BASE,strict=False)
    server.set_reference(ev,a.q,a.initial_pos,a.initial_quat)
    ev.reset_state_teleport();ev.obs_builder.reset();ev._ref_to_sim_ready=False
    ev._ensure_ref_to_sim_transform_rigid()
    trace=[];start=time.monotonic()
    for i in range(300):
        sample=i*2//5;stamp=sample*.05
        fraction=float(np.clip((stamp-1)/2,0,1))
        p=blend(BASE,target,fraction)
        q,pos,quat=a.update(p,dt=.02,sample_id=sample,timestamp=stamp)
        server.set_reference(ev,q,pos,quat);server.policy_step(ev)
        mujoco.mj_forward(ev.m,ev.d)
        entry=dict(sim_t_s=(i+1)*.02,pelvis_height_m=float(ev.d.qpos[2]),arms={})
        for side,(s,e,w) in [('left',(11,13,15)),('right',(12,14,16))]:
            arm=a.arms[side]
            wanted_u=unit(p[e,:3]-p[s,:3])@a.basis.T
            wanted_v=unit(p[w,:3]-p[e,:3])@a.basis.T
            angle=lambda u,v:float(np.degrees(np.arccos(np.clip(u@v,-1,1))))
            ref_u=unit(a.d.xpos[arm['elbow']]-a.d.xpos[arm['shoulder']])
            ref_v=unit(a.d.xpos[arm['wrist']]-a.d.xpos[arm['elbow']])
            actual_u=unit(ev.d.xpos[arm['elbow']]-ev.d.xpos[arm['shoulder']])
            actual_v=unit(ev.d.xpos[arm['wrist']]-ev.d.xpos[arm['elbow']])
            joints=[a.names[k] for k in arm['indices']]
            entry['arms'][side]=dict(reference_upper_error_deg=angle(wanted_u,ref_u),
                reference_fore_error_deg=angle(wanted_v,ref_v),
                actual_upper_error_deg=angle(wanted_u,actual_u),
                actual_fore_error_deg=angle(wanted_v,actual_v),
                actual_bend_deg=angle(actual_u,actual_v),
                reference_joint_deg=np.degrees(q[arm['indices']]).tolist(),
                command_joint_deg=[float(np.degrees(ev.target_dof_pos_by_name[n])) for n in joints],
                actual_joint_deg=[float(np.degrees(ev.d.qpos[int(ev.m.joint(n).qposadr[0])])) for n in joints])
        trace.append(entry)
        if ev.d.qpos[2]<.45:break
    report['cases'][name]=dict(steps=len(trace),wall_time_s=time.monotonic()-start,
        min_pelvis_height_m=min(t['pelvis_height_m'] for t in trace),
        fallen=bool(ev.d.qpos[2]<.45),final=trace[-1],trace=trace)
    (HERE/'policy-results.json').write_text(json.dumps(report,indent=2))
    print('POLICY_CASE',name,json.dumps({k:v for k,v in report['cases'][name].items() if k!='trace'}),flush=True)
