"""Offline URDF audit using one saved read-only state sample. No robot I/O."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def vec(node, key, default='0 0 0'):
    return np.fromstring(default if node is None else node.get(key, default), sep=' ')


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def transform(origin):
    r, p, y = vec(origin, 'rpy')
    t = np.eye(4)
    t[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    t[:3, 3] = vec(origin, 'xyz')
    return t


class Model:
    def __init__(self, path):
        self.path = path
        root = ET.parse(path).getroot()
        self.links = {l.get('name'): l for l in root.findall('link')}
        self.joints = {j.get('name'): j for j in root.findall('joint')}
        self.children = {n: [] for n in self.links}
        child_links = set()
        for name, j in self.joints.items():
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            self.children[parent].append((name, child))
            child_links.add(child)
        roots = set(self.links) - child_links
        assert len(roots) == 1, roots
        self.root = roots.pop()

    def descendants(self, link):
        return [link] + [n for _, child in self.children[link] for n in self.descendants(child)]

    def fk(self, q, root_rotation=None):
        first = np.eye(4)
        if root_rotation is not None:
            first[:3, :3] = root_rotation
        poses, axes, points = {self.root: first}, {}, {}

        def walk(parent):
            for name, child in self.children[parent]:
                j = self.joints[name]
                joint_frame = poses[parent] @ transform(j.find('origin'))
                axis = vec(j.find('axis'), 'xyz', '1 0 0')
                motion = np.eye(4)
                if j.get('type') in ('revolute', 'continuous'):
                    axes[name] = joint_frame[:3, :3] @ axis
                    points[name] = joint_frame[:3, 3]
                    motion[:3, :3] = rotation(axis, q.get(name, 0.))
                elif j.get('type') == 'prismatic':
                    motion[:3, 3] = axis * q.get(name, 0.)
                poses[child] = joint_frame @ motion
                walk(child)

        walk(self.root)
        return poses, axes, points

    def inertials(self, poses):
        result = {}
        for name, link in self.links.items():
            inertial = link.find('inertial')
            if inertial is None:
                continue
            mass = float(inertial.find('mass').get('value'))
            pose = poses[name] @ transform(inertial.find('origin'))
            e = inertial.find('inertia')
            matrix = np.array([[float(e.get('ixx')), float(e.get('ixy')), float(e.get('ixz'))],
                               [float(e.get('ixy')), float(e.get('iyy')), float(e.get('iyz'))],
                               [float(e.get('ixz')), float(e.get('iyz')), float(e.get('izz'))]])
            result[name] = (mass, pose[:3, 3], pose[:3, :3] @ matrix @ pose[:3, :3].T)
        return result

    def gravity(self, q, root_rotation):
        poses, axes, points = self.fk(q, root_rotation)
        inertials = self.inertials(poses)
        result = {}
        for name, axis in axes.items():
            child = self.joints[name].find('child').get('link')
            torque = np.zeros(3)
            for link in self.descendants(child):
                if link in inertials:
                    mass, com, _ = inertials[link]
                    torque += np.cross(com - points[name], [0, 0, -9.81 * mass])
            result[name] = -float(axis @ torque)
        return result

    def potential(self, q, root_rotation):
        return sum(m * 9.81 * com[2] for m, com, _ in self.inertials(self.fk(q, root_rotation)[0]).values())

    def group(self, q, links, relative_to):
        poses = self.fk(q)[0]
        data = self.inertials(poses)
        selected = [data[n] for n in links if n in data]
        mass = sum(m for m, _, _ in selected)
        com = sum(m * p for m, p, _ in selected) / mass
        inertia = np.zeros((3, 3))
        for m, p, i in selected:
            d = p - com
            inertia += i + m * ((d @ d) * np.eye(3) - np.outer(d, d))
        parent = poses[relative_to]
        com_local = parent[:3, :3].T @ (com - parent[:3, 3])
        i_local = parent[:3, :3].T @ inertia @ parent[:3, :3]
        return dict(mass_kg=mass, com_in_wrist_yaw_m=com_local.tolist(),
                    inertia_at_group_com_in_wrist_axes_kg_m2=i_local.tolist())


def run():
    state = json.loads((HERE / 'robot-state.json').read_text())
    q = dict(zip(state['names'], state['q']))
    q.update(zip(state['hand_names'], state['hand_q']))
    root_rotation = transform(ET.Element('origin', rpy=' '.join(map(str, state['imu_rpy']))))[:3, :3]
    models = {
        'holomotion_rubber': Model(ROOT / 'assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf'),
        'robot_xr_dex3': Model(HERE / 'g1_body29_hand14.urdf'),
    }
    report = dict(scope='software models and one measured pose; not hardware mass identification',
                  source_state_stamp=state['stamp'], mode_machine=state['mode_machine'], models={})
    torques = {}
    for key, model in models.items():
        limits = []
        for name, angle in q.items():
            j = model.joints.get(name)
            e = None if j is None else j.find('limit')
            if e is not None and 'lower' in e.attrib and not float(e.get('lower')) - .01 <= angle <= float(e.get('upper')) + .01:
                limits.append(dict(joint=name, angle_rad=angle, lower=float(e.get('lower')), upper=float(e.get('upper'))))
        torques[key] = model.gravity(q, root_rotation)
        finite_difference_errors = []
        for side in ('left', 'right'):
            for suffix in ('shoulder_pitch_joint', 'elbow_joint'):
                name = side + '_' + suffix
                plus, minus = dict(q), dict(q)
                plus[name] += 1e-6; minus[name] -= 1e-6
                numerical = (model.potential(plus, root_rotation) - model.potential(minus, root_rotation)) / 2e-6
                finite_difference_errors.append(abs(numerical - torques[key][name]))
        assert max(finite_difference_errors) < 1e-5
        hands = {}
        for side in ('left', 'right'):
            root = side + ('_rubber_hand' if key == 'holomotion_rubber' else '_hand_palm_link')
            hands[side] = model.group(q, model.descendants(root), side + '_wrist_yaw_link')
        report['models'][key] = dict(sha256=hashlib.sha256(model.path.read_bytes()).hexdigest(),
            hands=hands, measured_angles_outside_model_limits=limits,
            gravity_derivative_check_max_error_nm=max(finite_difference_errors))
    names = [n for n in state['names'] if 'shoulder_' in n or 'elbow_' in n or 'wrist_' in n]
    report['arm_gravity_at_saved_pose_nm'] = {
        n: dict(rubber=torques['holomotion_rubber'][n], dex3=torques['robot_xr_dex3'][n],
                difference=torques['robot_xr_dex3'][n] - torques['holomotion_rubber'][n]) for n in names}
    report['mounts'] = {}
    for side in ('left', 'right'):
        report['mounts'][side] = {}
        for key, model in models.items():
            j = model.joints[side + '_hand_palm_joint']
            report['mounts'][side][key] = dict(parent=j.find('parent').get('link'),
                child=j.find('child').get('link'), xyz=vec(j.find('origin'), 'xyz').tolist(),
                rpy=vec(j.find('origin'), 'rpy').tolist())
    report['standalone_hand_check'] = {}
    full = models['robot_xr_dex3']
    for side in ('left', 'right'):
        model = Model(HERE / f'unitree_dex3_{side}.urdf')
        data = model.inertials(model.fk(q)[0])
        differences = {}
        for name in sorted(set(model.links) & set(full.links)):
            a = model.links[name].find('inertial')
            b = full.links[name].find('inertial')
            if a is None or b is None:
                continue
            fields = {}
            for field in ('xyz', 'rpy'):
                x, y = vec(a.find('origin'), field), vec(b.find('origin'), field)
                if not np.allclose(x, y, atol=1e-12, rtol=0.):
                    fields['inertial_origin_' + field] = dict(standalone=x.tolist(), full_robot=y.tolist())
            for field in ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'):
                x, y = float(a.find('inertia').get(field)), float(b.find('inertia').get(field))
                if not np.isclose(x, y, atol=1e-12, rtol=0.):
                    fields[field] = dict(standalone=x, full_robot=y)
            if fields:
                differences[name] = fields
        common_joints = set(model.joints) & set(full.joints)
        joint_differences = []
        for name in sorted(common_joints):
            a, b = model.joints[name], full.joints[name]
            same = all(np.allclose(vec(a.find('origin'), field), vec(b.find('origin'), field), atol=1e-12, rtol=0.)
                       for field in ('xyz', 'rpy'))
            same = same and np.allclose(vec(a.find('axis'), 'xyz', '1 0 0'), vec(b.find('axis'), 'xyz', '1 0 0'))
            if not same:
                joint_differences.append(name)
        report['standalone_hand_check'][side] = dict(
            sha256=hashlib.sha256(model.path.read_bytes()).hexdigest(),
            mass_kg=sum(v[0] for v in data.values()), common_link_inertial_differences=differences,
            common_joint_geometry_differences=joint_differences)
    (HERE / 'audit-result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(hands={k: v['hands'] for k, v in report['models'].items()},
        standalone=report['standalone_hand_check'],
        gravity_delta_nm={n:v['difference'] for n,v in report['arm_gravity_at_saved_pose_nm'].items()}),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
