"""真实 HTTP/IK/ONNX/DDS/MuJoCo 集成测试；合成关键点，无真人测试声明。"""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2] if Path(__file__).resolve().parent.name=='tests' else Path('/home/lk/projects/HoloMotion')
BASE='http://127.0.0.1:8766'
def call(path,value=None):
    req=urllib.request.Request(BASE+path,data=None if value is None else json.dumps(value).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=10) as r: return json.load(r)

log=ROOT/'webcam_teleop/tests/integrated-dds.log'
log.parent.mkdir(exist_ok=True)
with log.open('w') as f:
    process=subprocess.Popen([sys.executable,'-u',str(ROOT/'webcam_teleop/server.py'),'--backend','dds','--port','8766'],stdout=f,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+150
        while time.monotonic()<deadline:
            if process.poll() is not None: raise RuntimeError('测试服务退出，检查 integrated-dds.log')
            try:
                s=call('/api/status')
                if s.get('error'): raise RuntimeError(s)
                if s['ready']: break
            except (OSError,TimeoutError): pass
            time.sleep(.5)
        else: raise RuntimeError('加载超时')
        # 复用摄像头链路的合成输入测试，不绕过 retarget 或 HTTP。
        script=(ROOT/'webcam_teleop/tests/synthetic_http.py').read_text()
        context={'__name__':'__main__','__file__':str(ROOT/'webcam_teleop/tests/synthetic_http.py')}
        exec(compile(script,'synthetic_http.py','exec'),context)
        before=call('/api/status'); assert before['backend']=='dds' and before['dds_pairs']>0
        with urllib.request.urlopen(BASE+'/frame.jpg',timeout=10) as r:
            assert r.read(2)==b'\xff\xd8'
        call('/api/control',{'action':'reset'}); time.sleep(.5)
        reset=call('/api/status'); assert not reset['running'] and not reset['calibrated'] and not reset['fallen'],reset
        # 验证中断接收端后，控制服务停止推进并公开错误，而不继续使用旧状态。
        context['pose']();call('/api/control',{'action':'calibrate'});time.sleep(.3)
        call('/api/control',{'action':'start'})
        children=[]
        for entry in Path('/proc').glob('[0-9]*'):
            try:
                status=(entry/'status').read_text()
                parent=int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))
                if parent==process.pid and b'dds_backend.py' in (entry/'cmdline').read_bytes(): children.append(int(entry.name))
            except (OSError,StopIteration): pass
        assert len(children)==1,children
        os.kill(children[0],signal.SIGTERM)
        for _ in range(40):
            context['pose']();time.sleep(.1); failed=call('/api/status')
            if failed.get('error') and not failed['running']: break
        else: raise AssertionError('DDS 中断未停止控制')
        assert 'DDS' in failed['error'],failed
        report=dict(backend='dds',dds_pairs=before['dds_pairs'],steps=before['steps'],reset_requires_calibration=True,jpeg_render=True,synthetic_http_test=True,dds_disconnect_stops=True,real_camera_tested=False)
        (ROOT/'webcam_teleop/tests/integrated-dds-results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
    finally:
        process.send_signal(signal.SIGINT)
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
