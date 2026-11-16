# HoloMotion Webcam + DDS 仿真原型

基于 Horizon Robotics HoloMotion 的本地摄像头上肢模仿扩展；新增可选 DDS 后端，将模型控制与 MuJoCo 物理计算拆成两个进程。

**本包是源码包，不包含预训练权重、Python 环境与完整训练环境，不能解压即运行。** 原始上游文档见 [README-upstream.md](README-upstream.md)。完整核心模型与上游工程由 Horizon Robotics 开发，保留原许可证、NOTICE 和作者声明。

## 从哪里开始

- [摄像头使用说明](webcam_teleop/README.md)
- [DDS 安装、运行与协议边界](webcam_teleop/DDS_README.md)
- [中文代码学习指南](webcam_teleop/代码学习指南.md)
- [待研发方案：骨架接入宇树官方遥操](待研发方案-骨架接入宇树官方遥操.md)
- [上游安装文档](README-upstream.md)
- 主要改动：webcam_teleop/server.py、retarget.py、dds_backend.py、web/、启动安装脚本和 tests/。

## 本机 WSL 挂载与运行位置

本机 Windows 的 `D:\dev\wsl-dev` 已挂载到 WSL `Ubuntu` 的 `/root/dev`，
因此本项目 `D:\dev\wsl-dev\HoloMotion-Webcam-DDS` 对应
`/root/dev/HoloMotion-Webcam-DDS`。两边是同一份文件，Windows 侧修改后直接在该 WSL
目录运行，不需要复制到另一份源码目录。可用 `findmnt -T /root/dev` 核对挂载。

本机现有解释器是 `/root/HoloMotion-Webcam-Source/.venv/bin/python`，当前安装的
ONNX Runtime 仅支持 CPU 推理；`HOLOMOTION_ALLOW_CPU=1` 用于显式允许 CPU 运行，
不能据此宣称达到 50 Hz 或等同 CUDA 实时效果。

纯 MuJoCo 动作调试入口为 8766，实机遥操入口为 8765；页面顶部可切换入口。
纯仿真不需要机器人连接、实机对齐或实机反馈：

```bash
cd /root/dev/HoloMotion-Webcam-DDS
bash webcam_teleop/start_sim.sh
```

Windows 浏览器访问 <http://localhost:8766>。
本机后台仿真日志为 WSL `/tmp/holomotion-sim-8766-*.log`。

## 运行概要

先准备 Ubuntu/WSL2、Python 3.11 和支持 CUDA 的 GPU，按上游说明安装能运行 HoloMotion 的依赖，并在工程根目录 `python -m pip install -e .`。
已测包版本见 webcam_teleop/requirements-tested.txt，仅作环境参考；其中本地编辑安装的第三方依赖要另按上游说明安装。

下载模型、机器人和 MediaPipe 资源，再生成初始化参考：

```bash
python -m pip install huggingface_hub
python webcam_teleop/download_resources.py
python webcam_teleop/prepare_first_demo.py
```

模型约 1.6 GB，需额外空间与网络，使用资源需遵守其原始许可。Git 子模块内容不随本包全部附带，来源见 .gitmodules；SDK 例外，源码与许可证保留在 webcam_teleop/vendor。
在确认原环境能运行后安装 DDS：

```bash
bash webcam_teleop/setup_dds.sh
bash webcam_teleop/start_dds.sh --port 8767
```

上述可选 DDS 仿真示例使用 http://localhost:8767 ，开启摄像头、校准、开始跟随。
Ctrl+C 停止服务。安装的系统编译依赖、CUDA/cuDNN 库路径要求见 DDS_README.md；全新机器端到端安装尚未验证。

直接模式使用上面的 `bash webcam_teleop/start_sim.sh`，固定 direct 后端和默认 8766 端口。

## 验证与限制

已测合成关键点 → HTTP → IK → 真实 HoloMotion CUDA 推理 → DDS → 独立 MuJoCo → 状态与 JPEG 反馈；并测试过期暂停、乱序拒绝、复位要求校准和 DDS 进程中断停控。
结果见 webcam_teleop/tests/integrated-dds-results.json 与 synthetic-results.json；不含用户摄像头录像。
DDS 初次集成实测峰值约 34 Hz，50 Hz 仅为目标；真人摄像头 DDS 模式尚待现场确认，原直接模式曾由用户确认抬臂跟随。

只支持站立上肢及小幅躯干跟随。DDS 命令为 Unitree LowCmd 类型，自定义会话主题、全状态反馈与执行器映射，并非实体机器人兼容实现，也不是原样的宇树官方 viewer。固定本机 lo 和 domain 73。

## 发布到 GitHub

解压后上传本目录中的文件，保留 LICENSE、NOTICE、README-upstream.md 和第三方 LICENSE。
推荐项目描述：“基于 HoloMotion 的摄像头上肢模仿与 DDS 本机仿真扩展”。不要把完整上游算法标为个人原创。
模型、环境、运行缓存、日志和本地编译产物已排除并写入 .gitignore。此包不包含 Git 历史，不会自动上传或提交。
