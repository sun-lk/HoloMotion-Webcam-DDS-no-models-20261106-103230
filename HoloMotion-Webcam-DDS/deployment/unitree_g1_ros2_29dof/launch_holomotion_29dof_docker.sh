#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROFILE="$SCRIPT_DIR/launch_profiles/orin_docker.yaml"

# 预检：本条是全身（行走 + 动作）通路，policy_node 强制加载双策略。
# 本包当前只验证上半身模式，未随包提供 velocity_tracking_model。
VELOCITY_DIR="$SCRIPT_DIR/src/models/velocity_tracking_model"
if [[ ! -f "$VELOCITY_DIR/config.yaml" ]] || ! compgen -G "$VELOCITY_DIR/exported/*.onnx" >/dev/null; then
    cat >&2 <<EOF
[launch] 已停止：缺少全身行走策略 velocity_tracking_model，无法启动全身模式。
  期望：$VELOCITY_DIR/{config.yaml,exported/*.onnx}
  来源：huggingface.co/HorizonRobotics/HoloMotion_models
  当前阶段仅支持上半身模式（站立上肢 + 小幅躯干，只加载 motion 模型）：
    cd $(cd "$SCRIPT_DIR/../../webcam_teleop" && pwd) && bash start.sh
  如需真机上半身跟随，请使用 g1-remote-action 桥（下肢锁站姿）。
EOF
    exit 1
fi

exec bash "$SCRIPT_DIR/scripts/launch_runtime.sh" \
    "$SCRIPT_DIR" \
    "$DEFAULT_PROFILE" \
    "HoloMotion 29DOF Docker" \
    "$0" \
    "$@"
