"""Read saved 1 Hz logs only. Does not import control code or access robot/network."""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (HERE / 'holomotion-motion-8765.jsonl').read_text().splitlines()]
current = [r for r in rows if r.get('bridge_flags', 0) & 32]
following = [r for r in current if r['phase'] == 'following']
names = following[0]['joint_names']
deg = 180 / np.pi
t = np.array([r['monotonic_s'] for r in following])
t -= t[0]
arrays = {key: np.array([r[key] for r in following]) * deg
          for key in ('reference_q_rad', 'command_q_rad', 'actual_q_rad', 'actual_dq_rad_s')}
ref, cmd, actual, velocity = (arrays[k] for k in arrays)
robot = []
for line in (HERE / 'robot-arm.log').read_text(encoding='utf-8').splitlines():
    if line.startswith('[status] '):
        try:
            robot.append(json.loads(line[9:]))
        except json.JSONDecodeError:
            pass  # An active logfile snapshot can end with an incomplete line.

joint_metrics = {}
for j in range(15, 29):
    joint_metrics[names[j]] = dict(
        reference_range_deg=float(np.ptp(ref[:, j])),
        upstream_command_range_deg=float(np.ptp(cmd[:, j])),
        actual_range_deg=float(np.ptp(actual[:, j])),
        upstream_command_vs_actual_max_difference_deg=float(np.max(np.abs(cmd[:, j] - actual[:, j]))),
        sampled_actual_speed_max_deg_s=float(np.max(np.abs(velocity[:, j]))))

delta = [r for r in robot if r['gravity']['reason'] == 'delta']
delta_window = [r for r in robot if delta[0]['t'] <= r['t'] <= delta[-1]['t']]
stale = [r for r in delta_window if r['reason'] == 'stale']
elbow = names.index('right_elbow_joint')
windows = []
for i in range(len(t)-4):
    sl = slice(i, i+5)
    if t[i+4]-t[i] <= 4.3 and np.ptp(ref[sl, elbow]) < 1:
        windows.append((float(np.ptp(cmd[sl, elbow])), i))
_, i = max(windows)
sl = slice(i, i+5)
quiet = dict(joint=names[elbow], start_s=float(t[i]), end_s=float(t[i+4]),
             reference_range_deg=float(np.ptp(ref[sl, elbow])),
             upstream_command_range_deg=float(np.ptp(cmd[sl, elbow])),
             actual_range_deg=float(np.ptp(actual[sl, elbow])))

offset = np.array([r['time_unix_s']-r['monotonic_s'] for r in following])
report = dict(
    local_total_records=len(rows), enabled_records=len(current), following_records=len(following),
    following_duration_monotonic_s=float(t[-1]),
    joint_metrics=joint_metrics, quiet_reference_joint_window=quiet,
    robot_records=len(robot), gravity_reason_counts=dict(Counter(r['gravity']['reason'] for r in robot)),
    sampled_stale_during_delta=[dict(robot_t_s=r['t'], last_action_seq=r['seq'],
                                   target_nm=r['gravity']['target_nm'], tau_nm=r['gravity']['tau_nm']) for r in stale],
    sampled_gravity_clipped_count=sum(r['gravity']['clipped'] for r in robot),
    sampled_invalid_gravity_state_count=sum(r['gravity']['reason'].startswith('invalid') for r in robot),
    max_local_feedback_age_ms=max(r['state_age_ms'] for r in current),
    wall_clock_offset_steps_s=[float(x) for x in np.diff(offset) if abs(x) > .1],
    limitations=[
        '1 Hz snapshots cannot resolve high frequency oscillation.',
        'command_q_rad is upstream of the robot bridge filter/slew, not final motor q.',
        'Human immobility is not established by these logs.',
        'Local wall clock stepped backwards; elapsed-time analysis uses monotonic_s.',
        'Robot t is relative to its own process; no exact cross-host alignment claimed.',
        'Timeout snapshots do not establish network, scheduling, or operator pause as the cause.',
    ])
(HERE / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout='constrained')
for ax, name, title in zip(axes, ['right_elbow_joint', 'left_wrist_roll_joint'],
                           ['Right elbow: reference, upstream command and measured angle',
                            'Left wrist roll: fixed reference with changing upstream command']):
    j = names.index(name)
    for values, label, color in [(ref, 'IK reference', '#777777'),
                                  (cmd, 'Upstream command', '#d17913'),
                                  (actual, 'Measured angle', '#176595')]:
        ax.plot(t, values[:, j], label=label, color=color, linewidth=1.7,
                linestyle='--' if label == 'IK reference' else '-')
    ax.set_title(title, loc='left')
    ax.set_ylabel('Joint angle (deg)')
    ax.grid(alpha=.15)
    ax.legend(loc='best', frameon=False, ncol=3, fontsize=9)
axes[0].axvspan(t[i], t[i+4], color='#777777', alpha=.12)
axes[0].text(.01, .97, f'Shaded 4 s: reference range {quiet["reference_range_deg"]:.2f} deg; '
             f'command range {quiet["upstream_command_range_deg"]:.2f} deg', transform=axes[0].transAxes,
             va='top', fontsize=9, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
axes[-1].set_xlabel('Elapsed time since first following sample (s, monotonic clock)')
fig.suptitle('Saved teleoperation logs — 2026-11-05, approx. 16:53 CST', fontsize=14)
fig.supxlabel('Source: local holomotion-motion-8765.jsonl | 38 samples, ~1 Hz | lines connect samples; fast motion is unresolved', fontsize=9)
fig.savefig(HERE / 'targets-vs-measured.png', dpi=160)
plt.close(fig)
print(json.dumps(dict(following_records=len(following), duration_s=t[-1], quiet=quiet,
                      delta_stale_records=len(stale), clock_steps=report['wall_clock_offset_steps_s']), indent=2))
