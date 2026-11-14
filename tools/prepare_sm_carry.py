#!/usr/bin/env python3
"""Prepare a supplied SM-s2r-proprio ONNX bundle; no simulator or hardware starts."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(q.shape[:-1]+(3,3))


def nominal_g1_profile(names, scales):
    """mjlab G1 10-Hz / damping-ratio-2 profile, checked against exported scales.

    Source: mujocolab/mjlab@61af2b75e9d44c93413aaa9bbace24575a9f6ad4,
    src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py.
    This is a SIMULATION profile; the bundle does not explicitly export nominal PD.
    """
    def reflected(rotor, g1, g2):
        return rotor[0]*(g1*g2)**2 + rotor[1]*g2**2 + rotor[2]
    motors = {
        '5020': (reflected((.139e-4,.017e-4,.169e-4),1+46/18,1+56/16),25.),
        '7520_14': (reflected((.489e-4,.098e-4,.533e-4),4.5,1+48/22),88.),
        '7520_22': (reflected((.489e-4,.109e-4,.738e-4),4.5,5),139.),
        '4010': (reflected((.068e-4,0,0),5,5),5.),
    }
    omega = 10*2*3.1415926535
    armature, effort = [], []
    for name in names:
        if 'wrist_pitch' in name or 'wrist_yaw' in name:
            key, factor = '4010',1
        elif 'ankle_' in name or name in ('waist_pitch_joint','waist_roll_joint'):
            key, factor = '5020',2
        elif 'hip_roll' in name or 'knee' in name:
            key, factor = '7520_22',1
        elif 'hip_pitch' in name or 'hip_yaw' in name or name == 'waist_yaw_joint':
            key, factor = '7520_14',1
        elif any(s in name for s in ('shoulder_', 'elbow_', 'wrist_roll')):
            key, factor = '5020',1
        else:
            raise ValueError(f'No motor profile for {name}')
        a,e = motors[key];armature.append(a*factor);effort.append(e*factor)
    armature, effort = np.array(armature),np.array(effort)
    kp = armature*omega**2;kd=4*armature*omega
    np.testing.assert_allclose(.25*effort/kp, scales, rtol=2e-6, atol=1e-8,
                               err_msg='Bundle action scales do not match the nominal G1 profile')
    return dict(kp=kp.tolist(),kd=kd.tolist(),armature=armature.tolist(),effort_limit=effort.tolist(),
                pd_provenance='mjlab nominal G1 profile cross-checked against exported action scale; not hardware verified',
                hardware_verified=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(bundle, reference, output):
    bundle,reference,output = Path(bundle).resolve(),Path(reference).resolve(),Path(output).resolve()
    contract = json.loads((bundle/'onnx-contract.json').read_text())
    validation = json.loads((bundle/'onnx-validation.json').read_text())
    assert contract['input']['shape'] == [1,733] and contract['output']['shape'] == [1,29]
    assert contract['reference']['offset_frames'] == list(range(0,50,5))
    assert math.isclose(contract['policy_dt_s'],.02) and contract['input']['history_length']==1
    assert sha(bundle/'actor.onnx') == validation['onnx_sha256'], 'ONNX checksum mismatch'
    with np.load(reference,allow_pickle=False) as data:
        names = data['joint_names'].tolist()
        assert names == contract['joint_names']
        assert data['body_names'].tolist()[0]=='pelvis' and math.isclose(float(data['fps']),50.)
        q=data['joint_pos'].astype(float);qd=data['joint_vel'].astype(float)
        quat=data['body_quat_w'][:,0].astype(float);pos=data['body_pos_w'][:,0].astype(float)
        root_vel=data['body_lin_vel_w'][0,0].astype(float);root_avel=data['body_ang_vel_w'][0,0].astype(float)
    for value in [q,qd,quat,pos,root_vel,root_avel]:assert np.isfinite(value).all()
    assert q.shape==qd.shape and q.shape[1]==29 and len(q)==len(quat)==len(pos)>0
    control=dict(contract,**nominal_g1_profile(names,contract['action_scale']))
    reference_data=dict(joint_names=names,fps=50.,offset_frames=list(range(0,50,5)),
                        joint_pos=q.tolist(),joint_vel=qd.tolist(),pelvis_quat_wxyz=quat.tolist())
    startup={'mujoco_sim_ros2_node':{'ros__parameters':{
        'startup_controllers':['state_estimator','walking_controller'],
        'initial_joint_names':names,'initial_joint_positions':q[0].tolist(),
        'initial_base_position':pos[0].tolist(),'initial_base_quaternion':quat[0].tolist()}}}
    rotations=rotation(quat)
    fixture_goals=[]
    for step in [0,len(q)//2,len(q)-1,len(q)+100]:
        current=rotations[min(step,len(q)-1)]
        idx=np.minimum(step+np.arange(10)*5,len(q)-1)
        goal=np.concatenate([np.concatenate([q[idx],qd[idx]],axis=-1).ravel(),
                             (current.T[None]@rotations[idx])[...,:2].ravel()])
        fixture_goals.append(dict(step=step,rotation=current.ravel().tolist(),goal=goal.tolist()))
    with np.load(bundle/'onnx-smoke-example.npz',allow_pickle=False) as smoke:
        fixtures=dict(obs=smoke['obs'].reshape(-1).tolist(),actions=smoke['actions'].reshape(-1).tolist(),goals=fixture_goals)
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('control.json',control),('reference.json',reference_data),('startup.yaml',startup),('fixtures.json',fixtures)]:
        (output/name).write_text(json.dumps(value,separators=(',',':'))+'\n')
    shutil.copy2(bundle/'actor.onnx',output/'actor.onnx')
    report=dict(source_reference=str(reference),reference_sha256=sha(reference),onnx_sha256=sha(output/'actor.onnx'),
                frames=len(q),duration_seconds=len(q)/50.,initial_root_velocity=root_vel.tolist(),
                initial_root_angular_velocity=root_avel.tolist(),initial_joint_velocity_norm=float(np.linalg.norm(qd[0])),
                initialization='Reference first pose, zero velocity held until controllers ready; not standby-to-policy validation',
                pd_provenance=control['pd_provenance'],physical_box=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();prepare(a.bundle,a.reference,a.output)
