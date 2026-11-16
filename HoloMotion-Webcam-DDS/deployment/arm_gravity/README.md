# 机器人端手臂重力前馈

2026-11-05 已部署至 `unitree@192.168.110.165:/home/unitree/dev/g1-remote-action`。
部署状态：`gravity.enabled: false`，未启动机器人控制桥，未做实机动作验证。
本机运行中的前端服务仍需重启才会加载新的阶段标记代码。

## 计算和指令

本机仍发送 29 维关节目标；原 ACTION reserved uint16 改为阶段标记，帧长
仍为 380 字节、协议版本仍为 1。0=旧客户端/未指定，1=对齐，2=策略跟随，
3=保持。旧桥忽略这个字段；新桥收到阶段 0 时前馈目标为零。
阶段只随通过校验的新序号动作更新，重复帧不能更改已接受的阶段。

机器人用 Pinocchio 固定基座模型和实际身体 29 关节、手指 14 关节计算重力。
通过骨盆 IMU 四元数将世界重力旋转到模型根坐标系，按关节名映射模型顺序。
只给 arm_sdk 的 15..28 电机写入前馈 tau；腰前馈仍为零。

- 对齐：采用现成 G1 + Dex3 URDF 的完整静态重力。
- HoloMotion 策略跟随：采用 Dex3 模型与原橡胶手模型的静态重力差量。
  这是考虑策略可能已承担原模型自重的首轮试验方案，尚未证实为最佳补偿量。
- 保持：只在动作持续有效时沿用最近有效的对齐/跟随阶段。启动时的保持不补偿。
- 动作超时、拒收、缺少阶段、身体/手指反馈无效时，前馈平滑退到零。
  手指缺失时不会用零角度代替；已测得手指角不会被 URDF 限位静默裁剪。

采用 `pin.computeGeneralizedGravity`，相当于零速度、零加速度下的逆动力学。
没有加入惯性、科氏力、摩擦辨识或自动负载估计；现成 Dex3 URDF 也不代表
已测出当前实物三指手的准确质量。

## 默认参数和保护

配置示例见 [gravity.example.yaml](gravity.example.yaml)。路径相对 bridge.yaml
所在目录解析，启用前先检查模型和依赖，再初始化任何 DDS 发布器。

模型计算 50 Hz，前馈力矩在 250 Hz 控制循环中限变化率。默认变化率
3 N·m/s、每次允许的 dt 不超过 20 ms，切换阶段也不跳变。
前馈限幅为配置额定力矩的 30%：当前肩肘与腕 roll 为 7.5 N·m，腕 pitch/yaw
为 1.5 N·m。这里只限制新增前馈，未改变原有 PD 控制的限幅语义。
身体反馈 100 ms、手指反馈 250 ms 过期时退到零。
退出回位期间也平滑撤掉前馈，再退出 arm_sdk 权重。

部署没有改动已有控制参数：肩肘 KP/KD=30/0.91，腕=30/0.91，腰=120/2。
`[status]` 日志新增 gravity 对象，包含阶段、原因、限幅标志、目标/实际施加前馈。
STATE bridge_flags 新增 0x10=支持阶段前馈、0x20=配置已启用；后者不表示此刻
一定存在非零前馈。前端 `/api/status` 和本机异步运动日志记录 bridge_flags。

## 验证和试机

机器人环境的 53 项离线测试全部通过，包括原协议/控制测试、阶段防重复更新、
前馈切换/超时/异常、SDK 电机映射和真实 Pinocchio 模型计算。
保存姿态下结果与独立 URDF 重力算法一致，并用势能有限差分核验力矩符号。
本机 39 项不依赖活动服务的单元测试通过。旧的 `test_integrated_dds.py` 是
导入即运行的独立集成脚本；本次全量发现测试时它因已有 8766 服务的骨架错误
退出，未作为本次功能验证结论，随后显式选择上述 39 项测试通过。

机器人纯双模型计算 1000 次，平均 0.260 ms、P95 0.412 ms；这不是 DDS 或
整个控制环的时序保证。实际输出见诊断目录的 deployment-result.json。

只读离线预检命令（不初始化 DDS、不发布动作）：

```bash
cd /home/unitree/dev/g1-remote-action
PYTHONPATH=src /home/unitree/miniconda3/envs/tv/bin/python -m g1_remote_action.arm_backend \
  --gravity-preflight --state-json tests/gravity_fixture/state.json
```

后续先重启本机前端加载阶段标记，再进行实机启用试验。
机器人端原启动命令增加 `--set gravity.enabled=true` 即可开启；省略时保持关闭。
启用试机前按项目规则取得操作员当次明确确认，先测固定人体姿态下的对齐保持，
核对 gravity.phase/reason/tau、实际姿态和下垂量，再测 full→delta 切换。
这些默认限幅与变化率尚未完成实机调参。

## 文件与回退

`overlay/` 保存相邻 `g1-remote-action` 工程的改动、现成模型和测试，便于在
HoloMotion 仓库内一起版本管理；它不是另一份活动运行目录。
`manifest.json` 记录部署文件哈希及部署前源码基线。

机器人部署前备份：
`/home/unitree/dev/g1-remote-action/gravity-backup-20261105-ub9cx2xj`。
其中 `config/bridge.yaml` 为未追加 gravity 的原配置，其他文件保持对应相对路径。
回退应在控制桥停止后进行：恢复备份中的原文件；新增模块/模型可保留未引用。
如仅关闭功能，移除启动参数 `--set gravity.enabled=true` 并保持 YAML enabled=false。
不要在运行中覆盖源码或配置。
