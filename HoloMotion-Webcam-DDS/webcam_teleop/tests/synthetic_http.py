"""Exercise real local HTTP -> retarget -> pretrained policy -> MuJoCo (synthetic landmarks)."""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np

BASE='http://127.0.0.1:8766'
def call(path, value=None):
    data=None if value is None else json.dumps(value).encode()
    req=urllib.request.Request(BASE+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=8) as r:
        return json.load(r)

def landmarks(raise_left=0):
    p=np.zeros((33,4));p[:,3]=.99
    for i,xyz in {11:[.2,-.5,0],12:[-.2,-.5,0],23:[.13,0,0],24:[-.13,0,0],
        13:[.22,-.23,0],15:[.22,.02,-.01],14:[-.22,-.23,0],16:[-.22,.02,-.01]}.items():p[i,:3]=xyz
    a=raise_left
    p[13,:3]=[.22+.24*a,-.23-.27*a,-.03*a]
    p[15,:3]=[.22+.48*a,.02-.52*a,-.04*a]
    return p.tolist()

call('/api/control',{'action':'reset'})
time.sleep(.2)
call('/api/control',{'action':'new_camera'})
seq=0
def pose(a=0):
    global seq
    r=call('/api/pose',{'sequence':seq,'landmarks':landmarks(a)});seq+=1;return r
pose();call('/api/control',{'action':'calibrate'});time.sleep(.2)
assert call('/api/status')['calibrated']
call('/api/control',{'action':'start'})
samples=[];start=time.monotonic()
while time.monotonic()-start<14:
    t=time.monotonic()-start
    a=0 if t<3 else min(1,(t-3)/3)
    if t>10:a=max(0,1-(t-10)/3)
    pose(a)
    if seq%10==0:
        s=call('/api/status')
        if 'pelvis_height' in s:
            samples.append(s)
        assert not s['fallen'],s
        assert s['error'] is None,s
    time.sleep(1/30)
before=call('/api/status');time.sleep(.8);stale=call('/api/status');time.sleep(.3);after=call('/api/status')
assert after['steps']==stale['steps'],(stale,after)
assert '过期' in after['stage'],after
call('/api/control',{'action':'pause'})
try:
    call('/api/pose',{'sequence':0,'landmarks':landmarks()})
    raise AssertionError('out of order frames accepted')
except urllib.error.HTTPError as exc:
    assert exc.code==400
report=dict(test='synthetic input; no real-camera claim',steps=before['steps'],
    min_pelvis_height=min(s['pelvis_height'] for s in samples),
    shoulder_reference_range=[min(s['left_shoulder_reference'] for s in samples),max(s['left_shoulder_reference'] for s in samples)],
    shoulder_actual_range=[min(s['left_shoulder_actual'] for s in samples),max(s['left_shoulder_actual'] for s in samples)],
    max_policy_hz=max(s['policy_hz'] for s in samples),stale_data_freeze=True,
    rejects_out_of_order=True,samples=samples)
assert report['shoulder_actual_range'][1]-report['shoulder_actual_range'][0]>.3,report
Path(__file__).resolve().parent.joinpath('synthetic-results.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
