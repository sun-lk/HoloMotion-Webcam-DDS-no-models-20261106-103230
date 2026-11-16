# 机器人端 URDF 对比（2026-11-05）

只读通过 SSH 从 `unitree@192.168.110.165` 下载模型，未运行机器人控制程序或修改运行配置。
完整结构化结果见 `comparison.json`。

## 来源与身份

- `robot_xr_g1_body29_hand14.urdf`：机器人 `/home/unitree/xr_teleoperate/assets/g1/g1_body29_hand14.urdf`。
  机器人上官方遥操的 `robot_arm_ik.py` 引用此路径；语音项目的同名模型 SHA256 与它相同。
- `g1_29dof_rev_1_0.urdf`：机器人 `/home/unitree/dev/HoloMotion-Webcam-DDS/assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf`。
  与本机项目同名 URDF 的 SHA256 完全相同。
- `g1_29dof_rev_1_0_s100.urdf`：同一机器人目录，增加 torso 上的固定 shell_support，质量 0.6 kg。

这些是机器人文件系统中的软件模型，并非现场硬件标定或实测结果。文件存在不能证明当前实机的版本、手部或负载与之相符。

## URDF 对比结果

- 本机与机器人 HoloMotion 标准 URDF 字节完全相同：29 个可动关节，所有惯性 link 质量合计 33.34114202 kg。
- 官方遥操 URDF 为 29 个身体关节加 14 个手指关节，质量合计 34.394234 kg。
- 两份 URDF 的共同 link，其质量、质心和惯量字段全部相同（浮点比较容差 abs=1e-8、rel=1e-7）。
- 29 个身体关节的父子结构、位置、旋转、轴方向和角度限位相同；肩、肘、腕的力矩和速度限制也相同。
- 官方遥操模型的四个踝关节和腰 roll/pitch，effort/velocity 为 50 Nm、37 rad/s；HoloMotion 对应值为 35 Nm、30 rad/s。
- HoloMotion 橡胶手每侧 0.17 kg；官方遥操模型掌部与手指 link 合计每侧 0.69654599 kg，增加约 0.526546 kg/侧。
  这是模型描述的差异，尚未确认实机是否装有该灵巧手。

## 当前 MuJoCo XML 另有差异

服务实际加载 `assets/robots/unitree/G1/29dof/scene_29dof.xml`，其包含 `g1_29dof.xml`。
显式 body inertial mass 合计 35.112142 kg，与训练配置引用的标准 URDF 合计相差约 1.771 kg。

为避免把固定零件合并当成质量变化，将 URDF 固定子 link 质量先合并到其活动父体后比较：

- torso：URDF 合并后 7.817 kg，当前 XML 9.598 kg，差 +1.781 kg。
- waist_yaw：0.214 kg 与 0.244 kg，差 +0.030 kg。
- waist_roll：0.086 kg 与 0.047 kg，差 -0.039 kg。
- pelvis：3.814 kg 与 3.813 kg，差 -0.001 kg。
- 两侧橡胶手质量已包含于 XML wrist_yaw body 中，不能重复相加。

固定合并比较在这里仅计算质量，没有比较合并后的惯量张量或质心；不能据此宣称完整动力学一致。
本次未替换 URDF、XML 或实机参数；这些结果不能单独确定手臂乱摆或跌倒的原因。
