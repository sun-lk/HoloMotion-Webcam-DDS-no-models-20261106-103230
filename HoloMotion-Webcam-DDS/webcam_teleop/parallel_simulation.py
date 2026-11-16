"""A separate MuJoCo state for preview and optional policy feedback.

One policy inference produces commands for both the robot and this simulation.
The simulation uses the model's existing PD/physics; no network or publishing.
"""
import mujoco
import numpy as np


class ParallelSimulation:
    def __init__(self, ev, control):
        self.ev = ev
        self.control = control
        self.measured_data = ev.d
        self.data = mujoco.MjData(ev.m)
        self.seed(ev.d)
        self.steps = 0
        self.last_targets = None
        self.error = None

    def seed(self, data):
        if data is not self.data:
            mujoco.mj_copyData(self.data, self.ev.m, data)
        self.error = None

    def select_feedback(self, source):
        if source not in ('robot', 'simulation'):
            raise ValueError('未知策略反馈来源')
        self.ev.d = self.data if source == 'simulation' else self.measured_data

    def advance(self, names, command, control_mode='arm_sdk'):
        """Arm targets match the sent frame; simulated legs retain policy balance.

        arm_sdk does not own real legs/waist. Its local preview still requires
        HoloMotion's full-body balance in the simulated environment.
        """
        if self.error:
            raise ValueError(f'本地仿真需要复位：{self.error}')
        ev = self.ev
        saved_data, saved_targets = ev.d, ev.target_dof_pos_by_name
        targets = dict(saved_targets)
        for name, q in zip(names, command):
            if control_mode == 'full_body' or any(part in name for part in ('shoulder', 'elbow', 'wrist')):
                targets[name] = float(q)
        try:
            ev.d = self.data
            ev.target_dof_pos_by_name = targets
            self.control(sleep=False)
            mujoco.mj_forward(ev.m, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise ValueError('本地仿真出现非有限状态，请复位')
            self.steps += 1
            self.last_targets = targets
            self.error = None
        except Exception as exc:
            self.error = str(exc)
            raise ValueError(f'本地仿真失败：{exc}') from exc
        finally:
            ev.d, ev.target_dof_pos_by_name = saved_data, saved_targets

    def snapshot(self, reference):
        if self.error:
            return None
        return self.data.qpos.copy(), self.data.qvel.copy(), np.asarray(reference).copy()

    def status(self):
        height = float(self.data.qpos[2])
        return dict(steps=self.steps, error=self.error,
                    pelvis_height=round(height, 3) if np.isfinite(height) else None,
                    fallen=bool(np.isfinite(height) and height < .45))


def choose_render_snapshot(source, simulation, robot):
    """No silent substitution: a missing robot frame must not look simulated."""
    if source == 'simulation':
        return simulation
    if source == 'robot':
        return robot
    raise ValueError('未知画面来源')
