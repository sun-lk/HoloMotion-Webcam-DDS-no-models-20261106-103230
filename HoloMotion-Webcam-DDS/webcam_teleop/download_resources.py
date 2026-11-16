"""Run from the activated Python environment; requires huggingface_hub."""
from pathlib import Path
import io, tarfile, urllib.request
from huggingface_hub import hf_hub_download, snapshot_download
ROOT = Path(__file__).resolve().parents[1]
model = ROOT / 'checkpoints'
for name in ['config.yaml', 'link_command_modes.json', 'exported/model_10500.onnx']:
    hf_hub_download('HorizonRobotics/HoloMotion_models', 'HoloMotion_motion_tracking_model_v2.0.0/'+name, local_dir=model)
snapshot_download('HorizonRobotics/HoloMotion_assets', repo_type='dataset', revision='795e5e1c1d3b6b15cc1cea9e7752742715724826', local_dir=ROOT/'thirdparties/HoloMotion_assets')
vendor = ROOT/'webcam_teleop/web/vendor'
vendor.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen('https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-1.1.1.tgz') as r:
    payload = r.read()
with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tf:
    for member in tf.getmembers():
        parts = Path(member.name).parts
        if len(parts)<2 or '..' in parts or not member.isfile(): continue
        target = vendor.joinpath(*parts[1:])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(tf.extractfile(member).read())
if not (vendor/'vision_bundle.js').exists():
    (vendor/'vision_bundle.js').write_bytes((vendor/'vision_bundle.cjs').read_bytes())
urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', vendor/'pose_landmarker_lite.task')
folder = ROOT/'data/first-demo'
folder.mkdir(parents=True, exist_ok=True)
hf_hub_download('lvhaidong/LAFAN1_Retargeting_Dataset', 'g1/walk1_subject1.csv', repo_type='dataset', revision='ce1572906efe6157840e8474d5a0d7aa87481e74', local_dir=folder)
(folder/'walk1_subject1.csv').write_bytes((folder/'g1/walk1_subject1.csv').read_bytes())
print('Resources downloaded. Next: python webcam_teleop/prepare_first_demo.py')
