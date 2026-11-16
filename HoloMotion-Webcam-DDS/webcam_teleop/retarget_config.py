"""retarget 参数加载：内置默认值 + YAML 覆盖，并对取值范围做校验。

原则：配置写错宁可启动就报错，也不要带着离谱的限幅跑起来——
这类错误在机器人上表现为「莫名其妙乱动」，现场很难反查。
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULTS = {
    "calibration": {
        "min_visibility": 0.5,
        "max_upper_arm_dev_deg": 35.0,
        "max_elbow_bend_deg": 30.0,
        "max_torso_tilt_deg": 15.0,
        "max_arm_asymmetry_deg": 20.0,
        "max_nominal_delta_rad": 0.45,
    },
    "mapping": {
        "waist_pitch_limit_rad": 0.18,
        "waist_roll_limit_rad": 0.18,
        "smoothing": 0.3,
        "max_rate_rad_s": 2.5,
        "ik_iterations": 5,
        "ik_damping": 0.004,
        "ik_step_limit_rad": 0.2,
        "joint_margin_rad": 0.02,
        "elbow_weight": 0.5,
        "ik_continuity_weight": 0.002,
        "ik_recovery_error_m": 0.03,
        "ik_recovery_iterations": 25,
    },
    "tracking": {
        "enabled": 1,
        "min_cutoff_hz": 1.0,
        "beta": 3.0,
        "derivative_cutoff_hz": 1.0,
        "max_landmark_speed_m_s": 3.0,
        "jump_margin_m": 0.03,
        "confidence_hold_s": 0.25,
        "reset_gap_s": 0.5,
        "hold_enter_rad": 0.005,
        "hold_exit_rad": 0.015,
        "hold_settle_s": 0.25,
    },
}

# (段, 键) -> 允许区间；越界直接拒绝启动。
_RANGES = {
    ("calibration", "min_visibility"): (0.0, 1.0),
    ("calibration", "max_upper_arm_dev_deg"): (1.0, 120.0),
    ("calibration", "max_elbow_bend_deg"): (1.0, 150.0),
    ("calibration", "max_torso_tilt_deg"): (1.0, 90.0),
    ("calibration", "max_arm_asymmetry_deg"): (1.0, 90.0),
    ("calibration", "max_nominal_delta_rad"): (0.01, 3.14),
    ("mapping", "waist_pitch_limit_rad"): (0.0, 1.57),
    ("mapping", "waist_roll_limit_rad"): (0.0, 1.57),
    ("mapping", "smoothing"): (0.01, 1.0),
    ("mapping", "max_rate_rad_s"): (0.01, 50.0),
    ("mapping", "ik_iterations"): (1, 50),
    ("mapping", "ik_damping"): (0.0001, 1.0),
    ("mapping", "ik_step_limit_rad"): (0.001, 1.57),
    ("mapping", "joint_margin_rad"): (0.0, 0.5),
    ("mapping", "elbow_weight"): (0.0, 10.0),
    ("mapping", "ik_continuity_weight"): (0.0, 0.1),
    ("mapping", "ik_recovery_error_m"): (0.005, 0.2),
    ("mapping", "ik_recovery_iterations"): (1, 100),
    ("tracking", "enabled"): (0, 1),
    ("tracking", "min_cutoff_hz"): (0.01, 30.0),
    ("tracking", "beta"): (0.0, 100.0),
    ("tracking", "derivative_cutoff_hz"): (0.01, 30.0),
    ("tracking", "max_landmark_speed_m_s"): (0.1, 30.0),
    ("tracking", "jump_margin_m"): (0.0, 0.5),
    ("tracking", "confidence_hold_s"): (0.0, 0.5),
    ("tracking", "reset_gap_s"): (0.1, 2.0),
    ("tracking", "hold_enter_rad"): (0.0, 0.1),
    ("tracking", "hold_exit_rad"): (0.0, 0.2),
    ("tracking", "hold_settle_s"): (0.0, 2.0),
}

_INT_FIELDS = {("mapping", "ik_iterations"), ("mapping", "ik_recovery_iterations"), ("tracking", "enabled")}


def default_path() -> Path:
    return Path(__file__).with_name("retarget_config.yaml")


def load(path=None) -> dict:
    """读取配置；path 为空时用同目录的 retarget_config.yaml（存在的话）。"""
    cfg = {section: dict(values) for section, values in DEFAULTS.items()}
    target = Path(path) if path else default_path()
    if target.is_file():
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"配置文件 {target} 顶层必须是映射")
        unknown_sections = set(data) - set(cfg)
        if unknown_sections:
            raise ValueError(f"未知配置段: {', '.join(sorted(unknown_sections))}")
        for section, block in data.items():
            if block is None:
                continue
            if not isinstance(block, dict):
                raise ValueError(f"配置段 {section} 必须是映射")
            for key, value in block.items():
                if key not in cfg[section]:
                    raise ValueError(f"未知配置项 {section}.{key}（拼写错误？）")
                cfg[section][key] = value
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    for (section, key), (low, high) in _RANGES.items():
        value = cfg[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{section}.{key} 必须是数字，当前 {value!r}")
        if not low <= float(value) <= high:
            raise ValueError(f"{section}.{key}={value} 超出允许范围 [{low}, {high}]")
        cfg[section][key] = int(value) if (section, key) in _INT_FIELDS else float(value)
    if cfg['tracking']['hold_exit_rad'] < cfg['tracking']['hold_enter_rad']:
        raise ValueError('tracking.hold_exit_rad 不得小于 hold_enter_rad')
    if cfg['tracking']['reset_gap_s'] < cfg['tracking']['confidence_hold_s']:
        raise ValueError('tracking.reset_gap_s 不得小于 confidence_hold_s')
