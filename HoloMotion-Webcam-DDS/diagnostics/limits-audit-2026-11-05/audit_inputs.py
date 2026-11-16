"""Offline examples of visibility veto and 2D/3D bone disagreement."""
from pathlib import Path
import json
import sys
import numpy as np
import mujoco
import yaml

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'webcam_teleop'))
from retarget import WebcamRetarget

r=json.loads((HERE/'ik-results.json').read_text())
robot=yaml.safe_load((ROOT/'checkpoints/HoloMotion_motion_tracking_model_v2.0.0/config.yaml').read_text())['robot']
names=robot['dof_names'];defaults=np.array(r['results']['arms_down_straight']['current']['q_reference'])
m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
base=np.zeros((33,4));base[:,3]=.99
for i,xyz in {11:[.2,-.5,0],12:[-.2,-.5,0],23:[.13,0,0],24:[-.13,0,0],
              13:[.2,-.23,0],15:[.2,.02,0],14:[-.2,-.23,0],16:[-.2,.02,0]}.items():base[i,:3]=xyz
side=base.copy();side[13,:3]=[.47,-.5,0];side[15,:3]=[.72,-.5,0]
side[14,:3]=[-.47,-.5,0];side[16,:3]=[-.72,-.5,0]
unit=lambda v:np.asarray(v)/np.linalg.norm(v)
angle=lambda u,v:float(np.degrees(np.arccos(np.clip(unit(u)@unit(v),-1,1))))
def solve(p):
 a=WebcamRetarget(m,names,robot['body_names'],defaults);a.calibrate(base,strict=False)
 for i in range(150):a.update(p,sample_id=i,timestamp=i*.05,dt=.05)
 return a

depth=[]
for shift in (0,.04,.08):
 p=side.copy();p[13,2]=p[14,2]=shift
 a=solve(p);arm=a.arms['left']
 depth.append(dict(illustrative_elbow_depth_error_m=shift,
     apparent_2d_bend_deg=angle(p[13,:2]-p[11,:2],p[15,:2]-p[13,:2]),
     input_3d_bend_deg=angle(p[13,:3]-p[11,:3],p[15,:3]-p[13,:3]),
     ik_bend_deg=angle(a.d.xpos[arm['elbow']]-a.d.xpos[arm['shoulder']],a.d.xpos[arm['wrist']]-a.d.xpos[arm['elbow']]),
     note='illustration, not measured MediaPipe error'))

a=solve(side);p=side.copy();p[15,3]=.49
held=[];errors=[]
for i in range(10):
 try:
  a.update(p,sample_id=150+i,timestamp=(150+i)*.05,dt=.05)
  held.append(round((i+1)*.05,3))
 except ValueError as e:errors.append(dict(elapsed_s=round((i+1)*.05,3),error=str(e)))

p=side.copy();p[[17,19,21],:3]=[[.6,-.4,.1],[.7,-.4,.2],[.6,-.5,.2]]
other=solve(p);plain=solve(side)
result=dict(depth_illustrations=depth,
            one_wrist_visibility_049=dict(held_elapsed_s=held,errors=errors),
            changing_hand_landmarks_only_max_reference_change_deg=float(np.max(np.abs(other.q-plain.q))*180/np.pi))
(HERE/'input-results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
