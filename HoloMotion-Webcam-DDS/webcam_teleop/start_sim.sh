#!/usr/bin/env bash
# Standalone simulation: never construct a G1RA/DDS robot transport.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -n "${HOLOMOTION_PYTHON:-}" ]]; then
  PYTHON="$HOLOMOTION_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "$HOME/HoloMotion-Webcam-Source/.venv/bin/python" ]]; then
  PYTHON="$HOME/HoloMotion-Webcam-Source/.venv/bin/python"
else
  PYTHON="$(command -v python || command -v python3)"
fi
# Renderer selected by server.py (WSLg/glfw or Linux/EGL).
unset HOLOMOTION_ALLOW_CPU

export HOLOMOTION_SIM_PD="${HOLOMOTION_SIM_PD:-model}"
# Final arguments intentionally prevent a caller from selecting a real backend.
exec "$PYTHON" -u -B webcam_teleop/server.py "$@" \
  --backend direct --port "${HOLOMOTION_SIM_PORT:-8766}" --pose-endpoint ''
