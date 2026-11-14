#!/usr/bin/env python3
"""Recompute runtime goals and feedback from recorded C++ controller inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
from prepare_sm_carry import rotation


def yaw(r):
    angle = np.arctan2(r[1, 0], r[0, 0])
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def validate(assets, trace):
    ref = json.loads((assets / 'reference.json').read_text())
    control = json.loads((assets / 'control.json').read_text())
    with np.load(trace, allow_pickle=False) as data:
        d = data['debug']
    assert d.ndim == 2 and d.shape[1] == 773 and len(d) > 1
    assert np.isfinite(d).all() and d[0, 0] == 0
    np.testing.assert_array_equal(np.diff(d[:, 0]), 1, err_msg='Missing policy frames')
    obs, actions = d[:, 11:744], d[:, 744:773]
    q, qd = np.array(ref['joint_pos']), np.array(ref['joint_vel'])
    rr, current = rotation(ref['pelvis_quat_wxyz']), rotation(d[:, 7:11])
    alignment = yaw(current[0]) @ yaw(rr[0]).T
    steps = d[:, 0].astype(int)
    indices = np.minimum(steps[:, None] + np.array(ref['offset_frames']), len(q)-1)
    relative = current.transpose(0, 2, 1)[:, None] @ alignment @ rr[indices]
    expected = np.concatenate([
        np.concatenate([q[indices], qd[indices]], axis=-1).reshape(len(d), 580),
        relative[..., :2].reshape(len(d), 60)], axis=-1)
    errors = dict(goal_max_error=float(np.abs(expected-obs[:, 93:]).max()),
                  gravity_max_error=float(np.abs(-current[:, 2, :]-obs[:, :3]).max()),
                  previous_action_max_error=float(np.abs(obs[1:, 64:93]-actions[:-1]).max()),
                  initial_previous_action_max_abs=float(np.abs(obs[0, 64:93]).max()))
    for name, value in errors.items():
        assert value < 1e-5, f'{name}: {value}'
    assert np.abs(actions).max() <= control['raw_action_limit']
    np.testing.assert_allclose(np.diff(d[:, 1]), .02, atol=1e-6)
    active = steps < len(q)
    actual_q = obs[:, 6:35] + np.array(control['default_joint_position_rad'])
    tracking_error = actual_q[active] - q[steps[active]]
    report = dict(interface_parity_passed=True, **errors,
                  joint_tracking_rmse_rad=float(np.sqrt(np.mean(tracking_error**2))),
                  joint_tracking_max_abs_rad=float(np.abs(tracking_error).max()),
                  policy_frames=len(d), completed_reference=bool(steps[-1] >= len(q)-1))
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    report = validate(a.assets, a.trace)
    if a.output:
        a.output.write_text(json.dumps(report, indent=2)+'\n')
