"""HTTP smoke test for the dedicated direct/MuJoCo service only."""
from pathlib import Path
import json
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'webcam_teleop/tests'))
from arm_pose_fixtures import BASE, POSES, blend

URL = 'http://127.0.0.1:8766'


def call(path, payload=None):
    request = urllib.request.Request(URL+path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


deadline = time.monotonic()+90
while True:
    state = call('/api/status')
    if state.get('error'):
        raise RuntimeError(state['error'])
    if state['ready']:
        break
    if time.monotonic() > deadline:
        raise TimeoutError('pure simulation startup timed out')
    time.sleep(.2)
assert state['backend'] == 'direct' and state['ik_solver'] == 'mink/daqp', state
assert not state['running'] and not state['robot_enabled'], 'An operator is using this service'
call('/api/control', {'action': 'new_camera'})
samples = []
try:
    call('/api/pose', {'sequence': 0, 'landmarks': BASE.tolist()})
    time.sleep(.15)
    call('/api/control', {'action': 'calibrate'})
    time.sleep(.15)
    call('/api/control', {'action': 'start'})
    start = time.monotonic()
    seq = 1
    while time.monotonic()-start < 10.:
        t = time.monotonic()-start
        if t < 3.:
            p = blend(BASE, POSES['side_raise_straight'], min(1., t/2))
        else:
            p = blend(POSES['side_raise_straight'], POSES['hello_forearms_up'], min(1., (t-3)/2))
        call('/api/pose', {'sequence': seq, 'timestamp_ms': t*1000, 'landmarks': p.tolist()})
        seq += 1
        if seq % 5 == 0:
            s = call('/api/status')
            assert not s['robot_enabled'] and s['backend'] == 'direct'
            assert not s['fallen'] and not s['error'], s
            samples.append(s)
        time.sleep(.05)
    final = call('/api/status')
    assert final['steps'] > 100, final
    image = urllib.request.urlopen(URL+'/frame.jpg', timeout=10).read()
    assert image[:2] == b'\xff\xd8'
    (HERE/'runtime-hello.jpg').write_bytes(image)
    report = dict(scope='synthetic HTTP + real ONNX + local MuJoCo only',
                  steps=final['steps'], ik_solver=final['ik_solver'],
                  min_pelvis_height_m=min(s['pelvis_height'] for s in samples),
                  fallen=final['fallen'], final_ik=final.get('ik'), samples=samples)
    (HERE/'runtime-results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'samples'}), flush=True)
finally:
    call('/api/control', {'action': 'pause'})
    call('/api/control', {'action': 'new_camera'})
    call('/api/control', {'action': 'reset'})
