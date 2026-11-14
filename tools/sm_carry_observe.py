#!/usr/bin/env python3
"""Record actual ROS controller observations/actions and simulation telemetry."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rclpy.init();n=Node('sm_carry_sim_observer');debug=[];ground=[];start=time.monotonic()
    trace=(out/'trace.jsonl').open('w')
    def receive(m):
        row=np.asarray(m.data,dtype=float)
        if len(row)!=773:raise RuntimeError(f'Unexpected debug message width {len(row)}')
        debug.append(row);trace.write(json.dumps({'type':'policy','values':row.tolist()})+'\n');trace.flush()
    def odom(m):
        pos=m.pose.pose.position;q=m.pose.pose.orientation
        row=[m.header.stamp.sec+m.header.stamp.nanosec*1e-9,pos.x,pos.y,pos.z,q.w,q.x,q.y,q.z]
        if not ground or row[0]-ground[-1][0]>=.035:
            ground.append(row);trace.write(json.dumps({'type':'mid360','values':row})+'\n')
    n.create_subscription(Float64MultiArray,'/sm_carry/debug',receive,100)
    n.create_subscription(Odometry,'/mid360',odom,qos_profile_sensor_data)
    try:
        while rclpy.ok() and time.monotonic()-start<120:
            rclpy.spin_once(n,timeout_sec=.1)
            if debug and debug[-1][0]>=debug[-1][3]+150:break
    finally:
        trace.close();d=np.asarray(debug);g=np.asarray(ground)
        result={'policy_samples':len(debug),'ground_truth_samples':len(ground),'wall_seconds':time.monotonic()-start,
                'scope':'Native ROS2 C++ controller in MuJoCo, reference-first-pose initialization, zero initial velocity; no physical box; no hardware test'}
        if len(d):
            obs=d[:,11:744];actions=d[:,744:773]
            result.update(first_step=int(d[0,0]),last_step=int(d[-1,0]),reference_frames=int(d[-1,3]),
                policy_elapsed_sim_seconds=float(d[-1,1]-d[0,1]),finite=bool(np.isfinite(d).all()),
                min_estimated_pelvis_height=float(d[:,6].min()),last_estimated_pelvis_height=float(d[-1,6]),
                max_abs_raw_action=float(np.abs(actions).max()),max_abs_joint_velocity=float(np.abs(obs[:,35:64]).max()),
                inference_ms={'median':float(np.median(d[:,2])),'p95':float(np.percentile(d[:,2],95)),'max':float(d[:,2].max())},
                previous_action_max_error=float(np.abs(obs[1:,64:93]-actions[:-1]).max()) if len(d)>1 else None)
            np.savez_compressed(out/'policy_trace.npz',debug=d,mid360=g)
        if len(g):result.update(min_mid360_height=float(g[:,3].min()),last_mid360_height=float(g[-1,3]),horizontal_sensor_displacement=float(np.linalg.norm(g[-1,1:3]-g[0,1:3])))
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
        n.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
