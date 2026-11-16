"""下载固定版本 DDS 原生源码并检查 SHA256，不下载运行模型。"""
import hashlib, io, tarfile, urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
url='https://codeload.github.com/eclipse-cyclonedds/cyclonedds/tar.gz/refs/tags/0.10.2'
with urllib.request.urlopen(url,timeout=120) as response: data=response.read()
assert hashlib.sha256(data).hexdigest()=='bc84e137e0c8a055b8cd97fbeafec94e36de1b0c2e88800896a82384fd867ae5','DDS 源码校验失败'
dest=root/'.local/src';dest.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive: archive.extractall(dest,filter='data')
print('CycloneDDS 0.10.2 source verified')
