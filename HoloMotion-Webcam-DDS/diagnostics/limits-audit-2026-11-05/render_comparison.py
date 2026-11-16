"""Static FK comparison of two branches, not a simulated balance trial."""
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL','osmesa')
import json
import mujoco
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
r=json.loads((HERE/'ik-results.json').read_text())
branches=json.loads((HERE/'bounded-branches.json').read_text())
robot=yaml.safe_load((ROOT/'checkpoints/HoloMotion_motion_tracking_model_v2.0.0/config.yaml').read_text())['robot']
names=robot['dof_names']
m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
q=np.array(r['results']['hello_forearms_up']['current']['q_reference'])
better=q.copy()
for side in ('left','right'):
    js=[f'{side}_{n}_joint' for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow')]
    values=branches['existing_solver_with_better_initial_branch'][side]['joint_deg']
    better[[names.index(n) for n in js]]=np.radians(values)
frames=[]
contacts=[]
camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(m,camera)
camera.distance=2.6;camera.azimuth=180;camera.elevation=-10;camera.lookat[:]=[0,0,.9]
with mujoco.Renderer(m,height=440,width=540) as renderer:
    for values in (q,better):
        d=mujoco.MjData(m);d.qpos[:3]=[0,0,.75];d.qpos[3:7]=[1,0,0,0]
        for n,value in zip(names,values):d.qpos[int(m.joint(n).qposadr[0])]=value
        mujoco.mj_forward(m,d)
        pairs=[]
        for contact in d.contact:
            bodies=[m.body(int(m.geom_bodyid[int(g)])).name for g in (contact.geom1,contact.geom2)]
            if 'world' not in bodies and any(any(k in n for k in ('shoulder','elbow','wrist')) for n in bodies):
                pairs.append(dict(bodies=bodies,distance_m=float(contact.dist)))
        contacts.append(pairs)
        renderer.update_scene(d,camera=camera);frames.append(renderer.render().copy())
fig,axes=plt.subplots(1,2,figsize=(11,5.2),layout='constrained')
for ax,frame,title in zip(axes,frames,['Current IK from normal initial pose','Same IK with a better initial branch']):
    ax.imshow(frame);ax.set_title(title,fontsize=12);ax.axis('off')
fig.suptitle('Same ideal "hello" input; original joint limits in both cases',fontsize=14)
fig.supxlabel('Static forward kinematics only. No balance / collision-clearance certification or hardware trial.',fontsize=9)
fig.savefig(HERE/'hello-ik-branches.png',dpi=150)
plt.close(fig)
(HERE/'static-arm-contacts.json').write_text(json.dumps(dict(current=contacts[0],better_initial_branch=contacts[1]),indent=2))
print(json.dumps(dict(current=contacts[0],better_initial_branch=contacts[1])))
