"""Mink 1.3.0 arm-position IK. No physics stepping or robot transport."""
import copy

import mink
import numpy as np


class MinkArmIK:
    """Fit elbows and wrists, holding every other degree of freedom exactly."""

    def __init__(self, model, arms, mapping):
        self.cfg = mapping
        self.arms = arms
        # Limit configuration belongs to IK only; never edit the physics model.
        self.model = copy.copy(model)
        self.active = np.array([v for arm in arms.values() for v in arm['vids']])
        self.qadr = np.array([self.model.jnt_qposadr[self.model.dof_jntid[v]]
                              for v in self.active])
        self.model.jnt_limited[:] = 0
        self.model.jnt_limited[self.model.dof_jntid[self.active]] = 1
        self.configuration = mink.Configuration(self.model)
        self.limits = [mink.ConfigurationLimit(
            self.model, min_distance_from_limits=mapping['joint_margin_rad']),
            mink.VelocityLimit(self.model, {
                self.model.joint(int(self.model.dof_jntid[v])).name:
                    mapping['ik_step_limit_rad'] for v in self.active})]
        frozen = sorted(set(range(model.nv)) - set(self.active))
        self.freeze = mink.DofFreezingTask(self.model, frozen)
        costs = np.zeros(model.nv)
        costs[self.active] = mapping['ik_continuity_weight']
        self.posture = mink.PostureTask(self.model, cost=costs)
        self.tasks = {}
        for side, arm in arms.items():
            for part, cost in [('elbow', mapping['elbow_weight']), ('wrist', 1.)]:
                self.tasks[side, part] = mink.FrameTask(
                    frame_name=arm[part], frame_type='body',
                    position_cost=cost, orientation_cost=0.)
        self.last_diagnostics = {}
        self.recovery_count = 0

    def _refine(self, qpos, goals, iterations):
        self.configuration.update(qpos)
        for _ in range(iterations):
            for key, task in self.tasks.items():
                # Keep target orientation equal to the current frame: the
                # SE(3) residual is then a pure point-position error. Webcam
                # landmarks do not supply a reliable palm orientation.
                rotation = self.configuration.get_transform_frame_to_world(
                    task.frame_name, 'body').rotation()
                task.set_target(mink.SE3.from_rotation_and_translation(rotation, goals[key]))
            try:
                velocity = mink.solve_ik(
                    self.configuration, [*self.tasks.values(), self.posture],
                    dt=1., solver='daqp', damping=self.cfg['ik_damping'],
                    limits=self.limits, constraints=[self.freeze], safety_break=True)
            except mink.exceptions.MinkError as exc:
                raise ValueError(f'Mink IK 求解失败，请重新校准：{exc}') from exc
            if not np.isfinite(velocity).all():
                raise ValueError('Mink IK 返回了无效关节速度，请重新校准')
            self.configuration.integrate_inplace(velocity, 1.)
        result = self.configuration.q.copy()
        if not np.isfinite(result).all():
            raise ValueError('Mink IK 返回了无效关节角，请重新校准')
        residuals = {
            f'{side}_{part}': float(np.linalg.norm(
                self.configuration.data.xpos[self.arms[side][part]] - goal))
            for (side, part), goal in goals.items()}
        return result, residuals

    def solve(self, qpos, goals):
        qpos = np.asarray(qpos).copy()
        margin = self.cfg['joint_margin_rad']
        bounds = np.concatenate([a['bounds'] for a in self.arms.values()])
        qpos[self.qadr] = np.clip(qpos[self.qadr], bounds[:, 0] + margin,
                                  bounds[:, 1] - margin)
        self.posture.set_target(qpos)
        result, residuals = self._refine(qpos, goals, int(self.cfg['ik_iterations']))
        best_error = max(residuals.values())
        selected = 'previous'
        # Local IK can stick to an elbow branch which would require exceeding
        # a real joint limit. Try only these three fixed, bounded seeds, and
        # only for arms whose geometric residual is large. All refinement is
        # still performed by Mink, never a second home-grown numerical solver.
        bad = [side for side in self.arms
               if max(residuals[f'{side}_{p}'] for p in ('elbow', 'wrist'))
               > self.cfg['ik_recovery_error_m']]
        if bad:
            candidates = [(result, residuals, 'previous')]
            seeds = [('elbow_up', (0., np.pi/2, np.pi/2, -.15)),
                     ('elbow_down', (0., np.pi/2, -np.pi/2, 1.4)),
                     ('forward', (-np.pi/2, 0., 0., -.15))]
            for name, angles in seeds:
                seed = result.copy()
                for side in bad:
                    arm = self.arms[side]
                    adr = self.model.jnt_qposadr[self.model.dof_jntid[arm['vids']]]
                    sign = 1 if side == 'left' else -1
                    seed[adr] = np.clip(np.asarray(angles) * [1, sign, sign, 1],
                                        arm['bounds'][:, 0] + margin,
                                        arm['bounds'][:, 1] - margin)
                try:
                    q, r = self._refine(seed, goals, int(self.cfg['ik_recovery_iterations']))
                except ValueError:
                    continue
                candidates.append((q, r, name))
            # First require a material fit improvement. Among similarly good
            # solutions choose the one closest to the previous reference, so
            # tiny residual differences cannot cause branch chatter.
            error = min(max(r.values()) for _, r, _ in candidates)
            if best_error - error > .005:
                near_best = [(q, r, name) for q, r, name in candidates
                             if max(r.values()) <= error + .003]
                result, residuals, selected = min(near_best, key=lambda x:
                    np.linalg.norm(x[0][self.qadr] - qpos[self.qadr]))
                self.recovery_count += 1
        # Restore the selected configuration (the last trial need not win).
        self.configuration.update(result)
        self.last_diagnostics = dict(solver='mink/daqp', seed=selected,
                                     recovery_count=self.recovery_count,
                                     residual_m=residuals)
        return result
