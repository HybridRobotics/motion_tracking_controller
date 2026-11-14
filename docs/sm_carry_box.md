# SuperMimic carry policy — NUC sim2sim

`sm_carry_box` adds `motion_tracking_controller/SmCarryController`, alongside the original
BeyondMimic controller. It reuses `legged_control2`'s ROS2 control, state estimator, joint
mapping, observation terms and ONNX CPU backend. Reference motion comes from external
NPZ data prepared into JSON, rather than the original controller's ONNX motion outputs.

## 当前验证范围

- Supplied bundle: `SM-s2r-proprio 2`, actor ONNX SHA256
  `3fb72f3cf52531728a79d7d670e75ff6b884cda24069779d4642a63fe3d7fada`.
- The bundle already includes the exported ONNX; deployment does not load `actor.pt`.
  Normalization and the motion encoder are inside the graph.
- Native C++ controller runs in MuJoCo on the NUC, at 50 Hz, with 500 Hz control updates.
- Simulation starts at reference frame 0 with **zero velocity**, held until both controllers
  are active. This is a policy tracking test, **not** a standby-to-policy transition test.
- Training bundle states that training did not include a physical box. These simulations
  contain **no physical box**. Arm motion following a carry reference does not verify
  lifting, contact stability, payload capacity, or actual box manipulation.
- PD gains are the nominal mjlab G1 10 Hz / damping-ratio-2 profile. All 29 exported action
  scales are checked against `0.25 * effort_limit / kp`. The bundle does not independently
  export nominal PD. This inference is documented and **not hardware verified**.
- `sm_carry_mujoco.launch.py` only creates a MuJoCo backend. There is no SM real launch
  in this branch. The original `real.launch.py` still targets the original controller.

## NUC 上的目录

Workspace: `/home/nuc-0326/sm_carry_box_ws`.

| Path under workspace | Contents |
| --- | --- |
| `src/motion_tracking_controller` | Fork checkout, branch `sm_carry_box` |
| `data/SM-s2r-proprio` | Supplied ONNX, contract, smoke fixture, reference NPZs |
| `assets_sm_carry/hold203` | Prepared hold reference, 281 frames / 5.62 s |
| `assets_sm_carry/full203` | Prepared full reference, 817 frames / 16.34 s |
| `results_sm_carry` | Actual recordings, logs, observations/actions and reports |

Weights, reference data and recordings are external assets excluded from Git. They remain
on the NUC; source, launch files, preparation and verification tools are versioned here.

## 本次 NUC 验证（2026-11-14）

| Test | Reference duration | Recorded policy duration | Joint tracking RMSE | Inference median / p95 |
| --- | --- | --- | --- | --- |
| `hold203_v1` | 5.62 s | 8.62 s | 0.0990 rad | 1.43 / 2.61 ms |
| `full203_v1` | 16.34 s | 19.34 s | 0.1150 rad | 1.51 / 3.12 ms |

Both recordings completed their reference plus end holding. Inspection of sampled video
frames showed squatting, standing and carry-pose tracking without falling; `full203`
also walked. This is a finite-duration observation, not a general stability claim.
Runtime goal/gravity parity errors were below `3e-15`; previous-action errors were zero.
The full test's simulated mid360 sensor moved 2.16 m horizontally between its first and
last sample. This is sensor endpoint displacement, not commanded forward distance or
the total path length. Full inference maximum was 11.05 ms, below the 20 ms policy period.

Machine-readable results are in [validation](validation/). Recordings and complete traces
are in the corresponding `~/sm_carry_box_ws/results_sm_carry/` directories on the NUC.

## Build / dependency setup

The tested NUC uses Ubuntu 24.04, ROS2 Jazzy, `legged_control2` 3.7.0 and MuJoCo 3.2.7.
The existing `~/colcon_ws` supplies Unitree description and the patched simulator.
Install the upstream dependencies in the repository README, plus:

```bash
sudo apt-get install nlohmann-json3-dev python3-numpy python3-yaml python3-pil ffmpeg xvfb
source /opt/ros/jazzy/setup.bash
source ~/colcon_ws/install/setup.bash
mkdir -p ~/sm_carry_box_ws/src
cd ~/sm_carry_box_ws/src
git clone -b sm_carry_box https://github.com/foursheepliu/motion_tracking_controller.git
cd ..
colcon build --symlink-install --allow-overriding motion_tracking_controller \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo --parallel-workers 1
```

This launch **requires** the startup-pose/controller-readiness extension in
`patches/mujoco_sim_ros2_startup.patch`. It is already installed on the test NUC.
On a fresh underlay, use `qiayuanl/mujoco_sim_ros2` commit
`088ae0f33e0a7e53b658627f03a66d9337aef650`, then apply and build:

```bash
git -C ~/colcon_ws/src/mujoco_sim_ros2 apply --check \
  ~/sm_carry_box_ws/src/motion_tracking_controller/patches/mujoco_sim_ros2_startup.patch
git -C ~/colcon_ws/src/mujoco_sim_ros2 apply \
  ~/sm_carry_box_ws/src/motion_tracking_controller/patches/mujoco_sim_ros2_startup.patch
cd ~/colcon_ws
colcon build --symlink-install --packages-select mujoco_sim_ros2
```

The patch waits for active `state_estimator` and `walking_controller`, applies the supplied
initial q/base pose, and prevents the robot from falling while controllers load. Its legacy
zero-velocity startup limitation is recorded in each prepared manifest. It also avoids
freeing shared MuJoCo data inside the SIGINT handler.

## Prepare and verify

Copy the supplied bundle into `~/sm_carry_box_ws/data/SM-s2r-proprio` first. Then:

```bash
cd ~/sm_carry_box_ws
python3 src/motion_tracking_controller/tools/prepare_sm_carry.py \
  --bundle data/SM-s2r-proprio \
  --reference data/SM-s2r-proprio/references/carry_full/carry_box_17-30-30_203_5792190_001.npz \
  --output assets_sm_carry/full203
source src/motion_tracking_controller/tools/sm_carry_env.sh
ros2 run motion_tracking_controller sm_carry_policy_probe "$PWD/assets_sm_carry/full203"
```

The preparation checks model SHA256, exact joint ordering, shapes, reference FPS, finite
values and the action-scale/PD relationship. It writes `actor.onnx`, `control.json`,
`reference.json`, `startup.yaml`, independent numerical fixtures and `manifest.json`.

The native probe compares C++ future windows against NumPy, ONNX actions against the
supplied smoke fixture, checks clipped previous actions and reset, and measures CPU latency.

## 在 NUC 桌面运行

Open a terminal in the NUC's graphical session:

```bash
bash ~/sm_carry_box_ws/src/motion_tracking_controller/tools/run_sm_carry_sim.sh \
  ~/sm_carry_box_ws/assets_sm_carry/full203
```

The NUC desktop also has **SM Carry Box 仿真**. To recreate that shortcut:

```bash
python3 ~/sm_carry_box_ws/src/motion_tracking_controller/tools/install_sm_carry_desktop.py
```

The simulator starts tracking automatically when controllers are ready; this simulation
does not use the real robot's remote. Close the window or Ctrl+C to finish. Do not run a
second simulator in the same ROS domain. Helpers use domain 72 and localhost discovery.
`SM_CARRY_WORKSPACE` and `SM_CARRY_UNDERLAY` override the workspace defaults.

## Record an isolated test

```bash
cd ~/sm_carry_box_ws
bash src/motion_tracking_controller/tools/record_sm_carry_sim.sh \
  assets_sm_carry/full203 results_sm_carry/full203_new
python3 src/motion_tracking_controller/tools/validate_sm_carry_trace.py \
  --assets assets_sm_carry/full203 \
  --trace results_sm_carry/full203_new/policy_trace.npz \
  --output results_sm_carry/full203_new/parity.json
```

Use a new result directory for every run. The recorder creates Xvfb `:93`, follows the
robot with a model camera, records the full reference plus approximately 3 seconds of
end-frame holding, and cleans up its processes. It saves `sim2sim.mp4`, `simulation.log`,
`trace.jsonl`, `policy_trace.npz`, and `result.json`. All movement is produced by the live
policy and MuJoCo physics; the recording is not a kinematic reference replay.

`/mid360` is simulated sensor ground truth at the sensor location, not the pelvis.
Estimated pelvis height is a separate metric. The trace validator independently checks
runtime goal construction, gravity projection, previous actions, reset, 50 Hz cadence and
action limits; joint tracking errors are reported separately from interface correctness.

## Model contract

Input is unnormalized `float32 obs[1,733]`; output is `actions[1,29]`:

| Input slice | Meaning |
| --- | --- |
| `0:3` | Body projected gravity |
| `3:6` | Body base angular velocity |
| `6:35` | Joint position minus exported defaults |
| `35:64` | Joint velocity; exported default velocity is zero |
| `64:93` | Previous post-clamp raw action, reset to zero |
| `93:673` | Ten frames, each `[29 q, 29 qdot]` |
| `673:733` | Ten relative pelvis rotations, six values each |

At policy step `t`, reference indices are `min(t + [0,5,...,45], N-1)`. Rotation is
`R_current_pelvis.T @ R_heading_alignment @ R_reference_pelvis`. Its first two columns
are flattened **row-major** as `[R00,R01,R10,R11,R20,R21]`. Initial heading alignment
preserves the reference yaw change while allowing a different starting heading.

Actions are clipped to `[-20,20]`, cached after clipping, and converted using
`q_desired = default_joint_position + action_scale * action`. Do not normalize twice,
substitute desired joint positions for previous raw actions, use the torso as the anchor,
or insert the original controller's time-step input.

At reference end, future lookup clamps to the last frame (including its stored velocities).
It does not loop and does not automatically switch into damping. Continued upright
behavior must be checked from the recording; an end-frame clamp is not a stability guarantee.

## Before a later sim2real adaptation

The remaining engineering work includes confirming actual training PD/model/contact
parameters, testing the standby-to-first-reference transition, integrating the SM plugin
into an explicitly reviewed real FSM, checking live state freshness and logging, and
separately validating payload/contact behavior. None of those hardware results is claimed
by the current sim2sim test.
