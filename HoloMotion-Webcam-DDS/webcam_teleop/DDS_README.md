# 摄像头 + HoloMotion + DDS 本机仿真

本模块在原摄像头上肢模仿程序中加入 `--backend dds`，并保留默认 `direct` 模式。
摄像头识别与人体动作映射不变；DDS 模式把物理推进交给独立进程。

```text
浏览器 MediaPipe → HTTP 人体关键点 → retarget → HoloMotion ONNX
  → LowCmd 类型的 DDS 电机指令 → 独立 MuJoCo 进程
  ← 自定义全状态 DDS 反馈 ← qpos / qvel
  → 本地状态镜像与画面渲染 → 浏览器
```

## 范围

- 本机仿真原型，仅 G1 29 关节；固定 `lo` 回环接口与 DDS 域 73。
- 每次启动使用独立会话主题 `rt/holomotion_sim/<session>/lowcmd` 和 `/state`，避免多个仿真互相驱动。
- 命令使用宇树 `unitree_hg::LowCmd_` 消息类型；35 个消息槽中仅使用 29 个，按当前 XML 执行器名称/顺序映射。
- 反馈是自定义 `holomotion_sim::SimState`，包含序号、qpos[36]、qvel[35]。它不是标准 `rt/lowstate`，不能宣称已兼容实体 G1。
- reserve[0] 用作锁步序号，reserve[1] 是本原型复位标记；这些是本机仿真协议约定。
- 独立进程使用原项目相同 XML、初始状态和步长；此后端不是未经修改直接启动的宇树官方 viewer。
- 当前支持站立上肢与小幅躯干动作，不支持行走、跳跃、深蹲、手指或精确全身动作捕捉。
- 不提供实体机器人连接入口。未来实机需要单独实现标准反馈、校验电机索引、低层状态机与现场保护。

## 运行条件与安装

先根据项目官方文档建立能运行原摄像头项目的 HoloMotion Python 3.11 环境，准备 GPU、ONNX 模型、G1 资源、初始参考 NPZ 和浏览器 MediaPipe 文件。
分享包中的 `download_resources.py`、`prepare_first_demo.py` 可下载/生成所需资源；完整说明见根目录 `README-分享与安装.md`。
已测环境版本清单见 requirements-tested.txt。它是版本记录，不是所有电脑通用的一键锁文件。完整 IsaacSim 训练环境不属于此演示要求。

Ubuntu 安装必要系统工具（需要管理员权限）：

```bash
sudo apt update
sudo apt install build-essential git libosmesa6 libgl1 libglib2.0-0
```

激活已经验证能运行 HoloMotion 的 Conda 环境，在工程根目录执行：

```bash
conda activate holomotion_train
bash webcam_teleop/setup_dds.sh
```

脚本创建 `.venv-dds`（继承已激活环境的包），构建 CycloneDDS 0.10.2 到 `.local/dds`，安装 Python 绑定和本包保留许可证的宇树 SDK 源码。
下载的 DDS 源码会校验 SHA256；编译依赖应先准备好。新建电脑安装全流程尚未验证，因此需要按报错检查 GPU、上游依赖和网络。

## 启动摄像头 DDS 服务

```bash
bash webcam_teleop/start_dds.sh
```

在 Windows Edge/Chrome 打开 http://localhost:8765 ，允许摄像头，校准，再开始跟随。
页面会显示“DDS 仿真”和反馈次数。摄像头图像在浏览器内处理；仅关键点传给本机服务，不默认录像。
Ctrl+C 停止；服务会结束独立仿真进程。DDS 进程被父进程回收，父进程意外退出时子进程也会检测并退出。

可配置 `HOLOMOTION_DDS_PYTHON` 使用已部署的独立解释器；默认使用工程 `.venv-dds/bin/python`。外部 DDS 原生库路径需通过 LD_LIBRARY_PATH 配置。
首次模型加载需要时间，页面可访问不等于 ready。不同端口可用 `--port 8766`；HTTP 来源检查已按实际端口适配。

直接控制模式在已激活原环境中运行：

```bash
python webcam_teleop/server.py --backend direct
```

## 暂停、复位与通信失败

- 未开始、未校准、关键点不可见或过期时，不发送新的物理子步；仿真接收端冻结物理时间。
- 每条命令等待同序号状态反馈，默认最多 2 秒。进程退出、反馈超时或非有限状态会停止控制并显示错误；DDS 通信故障后重启服务。
- 复位时同步两端状态，清除校准并要求重新校准。
- 序号旧包不执行；重复包返回状态，不重复推进物理。
- 执行器力矩按 XML ctrlrange 限幅；命令必须包含有限数值，kp/kd 不为负。
- 这些是仿真保护，不是实体机器人安全控制系统。

## 测试

```bash
# 先用 start_dds.sh 中同样的库路径环境；测试会占用 8766 端口并自行启停服务。
export LD_LIBRARY_PATH="$PWD/.local/dds/lib:${LD_LIBRARY_PATH:-}"
# 若 pip NVIDIA 库未加入路径，请按 start_dds.sh 导出库路径。
.venv-dds/bin/python webcam_teleop/tests/test_integrated_dds.py
```

测试通过 HTTP 发送合成肩肘腕髋关键点，经过实际 retarget、ONNX GPU、DDS 和独立 MuJoCo；检查动作响应、数据过期、旧帧拒绝、JPEG、复位和 DDS 接收端中断。
报告在 tests/integrated-dds-results.json 和 synthetic-results.json，日志在 integrated-dds.log。
这是合成输入的端到端验证；新 DDS 模式尚无真人摄像头现场效果确认。原 direct 模式曾由用户确认可以跟随抬臂。

实际频率取决于硬件和 DDS 往返延迟。本机首次集成测试最高约 33 Hz；50 Hz 是循环目标，不保证达到。

## 文件与来源

- server.py：接入后端选择、复位、通信异常和进程清理。
- dds_backend.py：DDS 控制端与独立仿真接收端，中文说明关键边界。
- start_dds.sh / setup_dds.sh / fetch_dds_source.py：启动、安装和源码校验。
- vendor/unitree_sdk2_python：宇树 SDK 源码，保留原 LICENSE。
- tests/：合成输入集成验证，不含用户摄像头录制。

本实现参考 https://github.com/unitreerobotics/unitree_mujoco 的 LowCmd PD 控制桥接方式，以及 https://github.com/unitreerobotics/unitree_sdk2_python 的消息与通道接口。
原 HoloMotion 代码保留 Apache-2.0 LICENSE、NOTICE 及作者声明。第三方代码按各自许可证使用。
