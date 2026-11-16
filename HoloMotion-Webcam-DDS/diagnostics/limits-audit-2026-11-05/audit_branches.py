"""Offline bounded multi-start IK: prove a pose can fit inside existing limits."""
from pathlib import Path
import json
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'webcam_teleop'))
from retarget import WebcamRetarget

r=json.loads((HERE/'ik-results.json').read_text())
m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
robot=yaml.safe_load((ROOT/'checkpoints/HoloMotion_motion_tracking_model_v2.0.0/config.yaml').read_text())['robot']
names=robot['dof_names']
base=np.array(r['results']['arms_down_straight']['current']['q_reference'])
a=WebcamRetarget(m,names,robot['body_names'],base)
normalize=lambda v:np.asarray(v)/np.linalg.norm(v)
angle=lambda u,v:float(np.degrees(np.arccos(np.clip(normalize(u)@normalize(v),-1,1))))
results={}
for name,upper,fore in [('hello_horizontal_upper',[0,1,0],[0,0,1]),
                       ('hello_diagonal_upper',[0,1,-1],[0,0,1]),
                       ('overhead_straight',[0,0,1],[0,0,1])]:
    by_side={}
    for side,sign in [('left',1),('right',-1)]:
        arm=a.arms[side];ix=arm['indices'];bounds=arm['bounds']
        u=normalize(upper)*[1,sign,1];v=normalize(fore)*[1,sign,1]
        def residual(x):
            q=base.copy();q[ix]=x;a.fk(q)
            au=normalize(a.d.xpos[arm['elbow']]-a.d.xpos[arm['shoulder']])
            av=normalize(a.d.xpos[arm['wrist']]-a.d.xpos[arm['elbow']])
            return np.r_[au-u,av-v]
        rng=np.random.default_rng(20261105)
        seeds=[base[ix], [0,sign*np.pi/2,sign*np.pi/2,-.1],
               [0,sign*np.pi/2,-sign*np.pi/2,1.4]]
        seeds+=list(rng.uniform(bounds[:,0]+.02,bounds[:,1]-.02,size=(10,4)))
        solutions=[]
        for seed in seeds:
            sol=least_squares(residual,np.clip(seed,bounds[:,0]+.020001,bounds[:,1]-.020001),
                              bounds=(bounds[:,0]+.02,bounds[:,1]-.02),
                              max_nfev=250,ftol=1e-11,xtol=1e-11,gtol=1e-11)
            solutions.append(sol)
        best=min(solutions,key=lambda s:np.linalg.norm(s.fun))
        q=base.copy();q[ix]=best.x;a.fk(q)
        au=a.d.xpos[arm['elbow']]-a.d.xpos[arm['shoulder']]
        av=a.d.xpos[arm['wrist']]-a.d.xpos[arm['elbow']]
        by_side[side]=dict(joint_deg=np.degrees(best.x).tolist(),upper_error_deg=angle(au,u),
                           fore_error_deg=angle(av,v),margin_to_model_limit_deg=np.degrees(np.minimum(best.x-bounds[:,0],bounds[:,1]-best.x)).tolist(),
                           good_seed_count=int(sum(np.linalg.norm(s.fun)<.05 for s in solutions)),seed_count=len(seeds))
    results[name]=by_side
    print(name,json.dumps(by_side),flush=True)

# Feed the existing production solver a better initial branch, keeping its
# objective, margin, damping and iteration count unchanged.
seed=results['hello_horizontal_upper']
p=np.zeros((33,4));p[:,3]=.99
p[11,:3],p[12,:3]=[.2,-.5,0],[-.2,-.5,0]
p[23,:3],p[24,:3]=[.13,0,0],[-.13,0,0]
cal=p.copy()
for s,e,w,sign in ((11,13,15,1),(12,14,16,-1)):
    cal[e,:3]=cal[s,:3]+[0,.27,0];cal[w,:3]=cal[e,:3]+[0,.25,0]
    p[e,:3]=p[s,:3]+[sign*.27,0,0];p[w,:3]=p[e,:3]+[0,-.25,0]
a=WebcamRetarget(m,names,robot['body_names'],base);a.calibrate(cal,strict=False)
for side in ('left','right'):
    a.q[a.arms[side]['indices']]=np.radians(seed[side]['joint_deg'])
a.reset_tracking()
for i in range(150):a.update(p,sample_id=i,timestamp=i*.05,dt=.05)
warm={}
for side,sign in [('left',1),('right',-1)]:
    arm=a.arms[side]
    warm[side]=dict(joint_deg=np.degrees(a.q[arm['indices']]).tolist(),
        upper_error_deg=angle(a.d.xpos[arm['elbow']]-a.d.xpos[arm['shoulder']],[0,sign,0]),
        fore_error_deg=angle(a.d.xpos[arm['wrist']]-a.d.xpos[arm['elbow']],[0,0,1]))
results['existing_solver_with_better_initial_branch']=warm
print('EXISTING_SOLVER_WARM_BRANCH',json.dumps(warm),flush=True)
(HERE/'bounded-branches.json').write_text(json.dumps(results,indent=2))
