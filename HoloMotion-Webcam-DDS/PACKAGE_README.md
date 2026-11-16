# 不含模型权重的项目快照

包含当前工作目录中的代码、配置、文档、诊断记录、URDF/MJCF、机器人网格及网页 JS/WASM；含尚未提交的 Mink 改动。

不包含 HoloMotion/ONNX/PyTorch 权重、MediaPipe .task 模型、SMPL/SMPL-X 参数、虚拟环境、Git 历史、缓存、日志和构建输出。排除清单及原有符号链接见 PACKAGE_CONTENTS.json。data/first-demo 是启动所用示例动作，予以保留。

机器人几何模型不属于神经网络权重，因此保留。此包不是开箱即用的完整运行环境，使用前需按项目说明补齐模型和依赖。g1-remote-action 同级项目不在这个项目目录内，未单独打入；deployment 中的远端补丁已保留。

这是 WSL 项目快照。建议在 WSL 内使用 unzip 解压以保留符号链接和脚本权限。清单中 target_exists_in_source=false 的链接在打包前就已经缺少目标，需要另行补齐相应外部资源。
