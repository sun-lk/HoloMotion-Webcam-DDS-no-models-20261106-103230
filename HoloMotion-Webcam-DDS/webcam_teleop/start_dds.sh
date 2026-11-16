#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${HOLOMOTION_DDS_PYTHON:-$ROOT/.venv-dds/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo '请先完成 HoloMotion 环境安装，并执行 bash webcam_teleop/setup_dds.sh。' >&2
  exit 1
fi
export MUJOCO_GL=osmesa
if [[ -d "$ROOT/.local/dds/lib" ]]; then
  export LD_LIBRARY_PATH="$ROOT/.local/dds/lib:${LD_LIBRARY_PATH:-}"
fi
# 同时兼容 pip 安装的 NVIDIA CUDA/cuDNN 动态库。
NVIDIA_LIBS="$($PYTHON -c 'import site,pathlib; print(":".join(str(p) for s in site.getsitepackages() for p in pathlib.Path(s).glob("nvidia/*/lib")))')"
if [[ -n "$NVIDIA_LIBS" ]]; then export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"; fi
cd "$ROOT"
exec "$PYTHON" -u webcam_teleop/server.py --backend dds "$@"
