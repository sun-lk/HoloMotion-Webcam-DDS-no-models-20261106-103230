import json
import hashlib
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
from holomotion.scripts.training.prepare_supermimic_lafan1 import resample_csv
from holomotion.src.env.reference_tracking.robot import G1_JOINT_NAMES

root = Path(__file__).resolve().parents[1]
folder = root / 'data/first-demo'
source = folder / 'walk1_subject1.csv'
raw = np.loadtxt(source, delimiter=',')[:301]
values = resample_csv(raw)
cfg = OmegaConf.load(root / 'checkpoints/HoloMotion_motion_tracking_model_v2.0.0/config.yaml')
joint_names = list(cfg.robot.dof_names)
body_names = list(cfg.robot.body_names)
joint_values = values[:, 7:][:, [G1_JOINT_NAMES.index(n) for n in joint_names]]
poses = values[:, :3].copy()
rotation = Rotation.from_quat(values[:, 3:7])
yaw = rotation[0].as_euler('xyz')[2]
heading = Rotation.from_euler('z', -yaw)
poses[:, :2] -= poses[0, :2]
poses = heading.apply(poses)
rotation = heading * rotation
model = mujoco.MjModel.from_xml_path(str(root / 'assets/robots/unitree/G1/29dof/scene_29dof.xml'))
data = mujoco.MjData(model)
qadr = [model.joint(n).qposadr[0] for n in joint_names]
body_ids = [model.body(n).id for n in body_names]
translations, quats = [], []
for t in range(len(values)):
    data.qpos[:3] = poses[t]
    xyzw = rotation[t].as_quat()
    data.qpos[3:7] = xyzw[[3, 0, 1, 2]]
    data.qpos[qadr] = joint_values[t]
    mujoco.mj_forward(model, data)
    translations.append(data.xpos[body_ids].copy())
    quats.append(data.xquat[body_ids][:, [1, 2, 3, 0]].copy())
translations = np.asarray(translations)
quats = np.asarray(quats)
angular = np.zeros_like(translations)
for b in range(len(body_ids)):
    r = Rotation.from_quat(quats[:, b])
    angular[:-1, b] = (r[1:] * r[:-1].inv()).as_rotvec() * 50
    angular[-1, b] = angular[-2, b]
arrays = dict(ref_dof_pos=joint_values, ref_dof_vel=np.gradient(joint_values, .02, axis=0),
    ref_global_translation=translations, ref_global_rotation_quat=quats,
    ref_global_velocity=np.gradient(translations, .02, axis=0), ref_global_angular_velocity=angular)
metadata = dict(motion_key='lafan1_walk1_subject1_first10s', motion_fps=50.,
    num_frames=len(values), num_dofs=29, num_bodies=30,
    source='lvhaidong/LAFAN1_Retargeting_Dataset',
    revision='ce1572906efe6157840e8474d5a0d7aa87481e74',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    conversion='First 10s; official 30-to-50Hz resampling; initial XY/yaw normalization; named-joint mapping; MuJoCo FK for G1 body poses')
out = folder / 'walk1_subject1_first10s.npz'
np.savez_compressed(out, metadata=json.dumps(metadata), **{k:v.astype(np.float32) for k,v in arrays.items()})
(folder / 'provenance.json').write_text(json.dumps(metadata, indent=2))
assert all(np.isfinite(v).all() for v in arrays.values())
print(out, json.dumps(metadata, indent=2))
