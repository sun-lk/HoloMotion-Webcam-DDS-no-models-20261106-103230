"""Noise-free bone directions for offline arm-IK regression tests."""
import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def pose(upper, fore):
    p = np.zeros((33, 4)); p[:, 3] = .99
    p[11, :3], p[12, :3] = [.2, -.5, 0], [-.2, -.5, 0]
    p[23, :3], p[24, :3] = [.13, 0, 0], [-.13, 0, 0]
    for (s, e, w), sign in [((11, 13, 15), 1), ((12, 14, 16), -1)]:
        def human(d):
            x, y, z = unit(d) * [1, sign, 1]
            return np.array([y, -z, -x])
        p[e, :3] = p[s, :3] + .27 * human(upper)
        p[w, :3] = p[e, :3] + .25 * human(fore)
    return p


BASE = pose([0, 0, -1], [0, 0, -1])
POSES = dict(side_raise_straight=pose([0, 1, 0], [0, 1, 0]),
             forward_raise_straight=pose([1, 0, 0], [1, 0, 0]),
             hello_forearms_up=pose([0, 1, 0], [0, 0, 1]))


def blend(start, end, t):
    p = start.copy()
    for s, e, w in ((11, 13, 15), (12, 14, 16)):
        for parent, child, length in ((s, e, .27), (e, w, .25)):
            u = unit(start[child, :3] - start[parent, :3])
            v = unit(end[child, :3] - end[parent, :3])
            dot = float(np.clip(u @ v, -1, 1))
            if dot > .99999:
                direction = unit((1-t)*u + t*v)
            elif dot < -.99999:
                axis = np.array([0., 0., 1.])
                if abs(axis @ u) > .9:
                    axis = np.array([1., 0., 0.])
                axis = unit(axis - (axis @ u) * u)
                direction = np.cos(np.pi*t)*u + np.sin(np.pi*t)*axis
            else:
                angle = np.arccos(dot)
                direction = (np.sin((1-t)*angle)*u + np.sin(t*angle)*v)/np.sin(angle)
            p[child, :3] = p[parent, :3] + length*direction
    return p


def errors(adapter, p, data=None):
    d = adapter.d if data is None else data
    result = {}
    angle = lambda u, v: float(np.degrees(np.arccos(np.clip(unit(u) @ unit(v), -1, 1))))
    for side, (s, e, w) in [('left', (11, 13, 15)), ('right', (12, 14, 16))]:
        arm = adapter.arms[side]
        upper = d.xpos[arm['elbow']] - d.xpos[arm['shoulder']]
        fore = d.xpos[arm['wrist']] - d.xpos[arm['elbow']]
        result[side] = dict(
            upper_deg=angle(upper, (p[e, :3]-p[s, :3]) @ adapter.basis.T),
            fore_deg=angle(fore, (p[w, :3]-p[e, :3]) @ adapter.basis.T),
            bend_deg=angle(upper, fore))
    return result
