#!/usr/bin/env bash
# HoloMotion(开发机) -> g1-remote-action(机器人) 适配器启动脚本。
# 依赖：ROS2 环境 + unitree_hg/unitree_go 消息包 + g1-remote-action 的 protocol 模块。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# g1-remote-action 的 src 路径（开发机上的部署位置），可用 G1RA_HOME 覆盖
G1RA_HOME="${G1RA_HOME:-$HOME/dev/g1-remote-action}"
# 优先用 colcon 安装后的包；未 build 时回退到源码目录里的 humanoid_policy
export PYTHONPATH="$G1RA_HOME/src:$ROOT/src:${PYTHONPATH:-}"

if [[ -f "$ROOT/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/install/setup.bash"
fi

: "${ROS_DISTRO:=${ROS_DISTRO:-humble}}"
if [[ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "/opt/ros/$ROS_DISTRO/setup.bash"
fi

# 预检：缺依赖时给出可执行的提示，而不是等到 import 阶段抛栈
if [[ ! -d "$G1RA_HOME/src/g1_remote_action" ]]; then
  echo "[g1ra] 已停止：找不到 g1-remote-action 的 protocol 模块（$G1RA_HOME/src/g1_remote_action）。" >&2
  echo "       请把 g1-remote-action 放到 $G1RA_HOME，或用 G1RA_HOME=<路径> 覆盖。" >&2
  exit 1
fi
if ! python3 - <<'PY'
import importlib, sys
missing = []
for mod in ("rclpy", "unitree_hg", "unitree_go", "std_msgs", "zmq", "numpy"):
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
if missing:
    print("[g1ra] 缺少 Python 依赖: " + ", ".join(missing)
          + "（需 ROS2 + unitree_ros2 已 source，以及 pyzmq/numpy）", file=sys.stderr)
    sys.exit(1)
PY
then
  echo "[g1ra] 已停止：适配器依赖不完整（见上）。" >&2
  echo "       ROS 侧先 source /opt/ros/\$ROS_DISTRO/setup.bash 与 unitree_ros2/setup.sh。" >&2
  exit 1
fi

echo "[g1ra] 上半身模式：只转发 /humanoid/action，下肢由机器人侧桥锁定在站姿"
exec python3 -m humanoid_policy.g1ra_adapter_node "$@"
