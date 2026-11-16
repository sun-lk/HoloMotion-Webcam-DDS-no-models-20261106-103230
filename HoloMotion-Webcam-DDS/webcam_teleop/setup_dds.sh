#!/usr/bin/env bash
# 先激活已经能运行 HoloMotion 的 Python 3.11 环境，再运行此脚本。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${HOLOMOTION_PYTHON:-python}"
"$BASE_PYTHON" -c 'import sys, mujoco, torch, onnxruntime; assert sys.version_info[:2] == (3,11), "请使用 Python 3.11"; print(sys.executable)'
"$BASE_PYTHON" -m venv --system-site-packages "$ROOT/.venv-dds"
PYTHON="$ROOT/.venv-dds/bin/python"
"$PYTHON" -m pip install 'cmake==4.4.4'
"$PYTHON" "$ROOT/webcam_teleop/fetch_dds_source.py"
"$ROOT/.venv-dds/bin/cmake" -S "$ROOT/.local/src/cyclonedds-0.10.2" -B "$ROOT/.local/dds-build" \
  -DCMAKE_INSTALL_PREFIX="$ROOT/.local/dds" -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
  -DENABLE_SECURITY=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
"$ROOT/.venv-dds/bin/cmake" --build "$ROOT/.local/dds-build" -j 4
"$ROOT/.venv-dds/bin/cmake" --install "$ROOT/.local/dds-build"
CYCLONEDDS_HOME="$ROOT/.local/dds" "$PYTHON" -m pip install 'cyclonedds==0.10.2'
"$PYTHON" -m pip install --no-deps -e "$ROOT/webcam_teleop/vendor/unitree_sdk2_python"
echo 'DDS 环境已安装。启动：bash webcam_teleop/start_dds.sh'
