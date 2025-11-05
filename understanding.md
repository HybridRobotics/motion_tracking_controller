# Motion Tracking Controller - Technical Deep Dive

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Breakdown](#component-breakdown)
4. [Robot API Access](#robot-api-access)
5. [Observation System](#observation-system)
6. [ONNX Model Integration](#onnx-model-integration)
7. [Data Flow](#data-flow)
8. [Configuration](#configuration)
9. [Coordinate Frames and Transformations](#coordinate-frames-and-transformations)

---

## Overview

The Motion Tracking Controller is a ROS 2 Humble controller that enables humanoid robots (specifically Unitree G1) to track and execute motion trajectories generated from neural network policies. It's built on the **legged_control2** framework and uses ONNX runtime for efficient neural network inference.

**Key Purpose**: Bridge the gap between trained motion tracking policies (from reinforcement learning) and real robot execution by:
- Loading ONNX models with embedded motion trajectories
- Computing observations aligned with the training environment
- Tracking reference motions in real-time
- Providing joint position/velocity commands to the robot

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    ROS 2 Control Framework                   │
│                    (controller_manager)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              MotionTrackingController                        │
│              (inherits from RlController)                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  CommandManager         ObservationManager            │  │
│  │  ├─ MotionCommandTerm   ├─ MotionAnchorPosition      │  │
│  │  └─ (other terms)       ├─ MotionAnchorOrientation   │  │
│  │                          ├─ RobotBodyPosition         │  │
│  │                          └─ RobotBodyOrientation      │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│              ┌──────────────────────┐                        │
│              │  MotionOnnxPolicy    │                        │
│              │  (ONNX Runtime)      │                        │
│              └──────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Robot Hardware Interface                        │
│              (Unitree SDK / MuJoCo Sim)                      │
└─────────────────────────────────────────────────────────────┘
```

### Class Hierarchy

```
controller_interface::ControllerInterface (ROS 2 base)
    └── RlController (legged_control2 framework)
        └── MotionTrackingController (this package)
            ├── Uses: MotionOnnxPolicy
            ├── Uses: MotionCommandTerm
            └── Uses: MotionObservation* classes
```

---

## Component Breakdown

### 1. MotionTrackingController (`MotionTrackingController.h/cpp`)

**Location**: `include/motion_tracking_controller/MotionTrackingController.h:9`

**Purpose**: Main controller class that orchestrates the motion tracking pipeline.

**Key Responsibilities**:
- Initialize and configure the ONNX policy
- Parse and register command/observation terms
- Inherit ROS 2 lifecycle management from `RlController`

**Lifecycle Methods**:

```cpp
on_init()      // Declare parameters (motion.start_step)
on_configure() // Load ONNX model, extract metadata (anchor body, body names)
on_activate()  // Start controller execution
on_deactivate()// Stop controller
```

**Custom Parsers** (`MotionTrackingController.cpp:53-81`):
- `parserCommand()`: Registers "motion" command term
- `parserObservation()`: Registers motion-specific observations:
  - `motion_ref_pos_b` / `motion_anchor_pos_b` → MotionAnchorPosition
  - `motion_ref_ori_b` / `motion_anchor_ori_b` → MotionAnchorOrientation
  - `robot_body_pos` → RobotBodyPosition
  - `robot_body_ori` → RobotBodyOrientation

---

### 2. MotionOnnxPolicy (`MotionOnnxPolicy.h/cpp`)

**Location**: `include/motion_tracking_controller/MotionOnnxPolicy.h:11`

**Purpose**: Wraps ONNX neural network inference and extracts reference motion trajectories.

**Key Features**:

**Inputs to ONNX Model**:
- `observations`: Current robot state observations (from ObservationManager)
- `time_step`: Current timestep in the motion sequence

**Outputs from ONNX Model** (`MotionOnnxPolicy.cpp:23-38`):
- `joint_pos`: Reference joint positions (num_joints,)
- `joint_vel`: Reference joint velocities (num_joints,)
- `body_pos_w`: Reference body positions in world frame (num_bodies, 3)
- `body_quat_w`: Reference body orientations in world frame (num_bodies, 4) - wxyz format

**Metadata Parsing** (`MotionOnnxPolicy.cpp:43-49`):
The ONNX file contains embedded metadata:
- `anchor_body_name`: The body frame used as reference (e.g., "pelvis")
- `body_names`: List of tracked body names (e.g., ["left_hand", "right_hand", "pelvis"])

**Time Stepping**:
```cpp
timeStep_ = startStep_;  // Initialize from parameter (default 0)
timeStep_++;             // Increment each forward() call
```

This allows starting playback from a specific frame in the motion sequence.

---

### 3. MotionCommandTerm (`MotionCommand.h/cpp`)

**Location**: `include/motion_tracking_controller/MotionCommand.h:16`

**Purpose**: Provides reference joint commands (position + velocity) from the ONNX policy.

**getValue() Output** (`MotionCommand.cpp:9-11`):
Returns a concatenated vector of size `2 * num_joints`:
```
[joint_pos_1, ..., joint_pos_N, joint_vel_1, ..., joint_vel_N]
```

**Frame Alignment** (`MotionCommand.cpp:36-42`):

On `reset()`, the controller aligns the motion trajectory with the robot's current pose:

1. Extract anchor body pose from motion: `initToAnchor`
2. Extract anchor body pose from robot: `worldToAnchor`
3. Only use yaw orientation (project to ground plane)
4. Compute alignment transform: `worldToInit_ = worldToAnchor * initToAnchor.inverse()`

This ensures the motion starts from the robot's current position and orientation.

**Helper Functions**:
- `getAnchorPositionLocal()`: Returns motion anchor position in robot's local frame
- `getAnchorOrientationLocal()`: Returns motion anchor orientation as 6D rotation matrix
- `getRobotBodyPositionLocal()`: Returns robot body positions relative to anchor
- `getRobotBodyOrientationLocal()`: Returns robot body orientations relative to anchor

---

### 4. MotionObservation Classes (`MotionObservation.h`)

**Location**: `include/motion_tracking_controller/MotionObservation.h:14`

These classes compute specific observation terms fed to the neural network policy.

**Base Class**: `MotionObservation` (line 14)
- Holds reference to `MotionCommandTerm` for accessing motion data

**Derived Observation Terms**:

#### MotionAnchorPosition (line 22)
- **Size**: 3 (x, y, z)
- **Calculation**: `commandTerm_->getAnchorPositionLocal()`
- **Meaning**: Position of the motion anchor in robot's anchor frame

#### MotionAnchorOrientation (line 31)
- **Size**: 6 (2 columns of 3x3 rotation matrix)
- **Calculation**: `commandTerm_->getAnchorOrientationLocal()`
- **Meaning**: Orientation of motion anchor as 6D rotation representation
- **Format**: [R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]]

#### RobotBodyPosition (line 40)
- **Size**: 3 * num_bodies
- **Calculation**: `commandTerm_->getRobotBodyPositionLocal()`
- **Meaning**: Positions of all tracked robot bodies in anchor frame
- **Format**: [body1_x, body1_y, body1_z, body2_x, ...]

#### RobotBodyOrientation (line 49)
- **Size**: 6 * num_bodies
- **Calculation**: `commandTerm_->getRobotBodyOrientationLocal()`
- **Meaning**: Orientations of all tracked robot bodies in anchor frame
- **Format**: 6D rotation representation for each body

---

## Robot API Access

### Through legged_control2 Framework

The controller doesn't directly access robot hardware. Instead, it uses the **legged_control2** framework abstractions:

#### 1. **Model Access** (`model_` member from RlController)

**Pinocchio Robot Model**: Provides kinematic/dynamic computations
```cpp
const auto& pinModel = model_->getPinModel();
const auto& pinData = model_->getPinData();
```

**Available APIs**:
- `pinModel.getFrameId(name)`: Get frame index by name
- `pinModel.nframes`: Total number of frames
- `pinData.oMf[index]`: SE3 pose of frame in world coordinates
- `model_->getNumJoints()`: Get number of actuated joints

**Example** (`MotionCommand.cpp:14-24`):
```cpp
anchorRobotIndex_ = pinModel.getFrameId(cfg_.anchorBody);
pinocchio::SE3 worldToAnchor = model_->getPinData().oMf[anchorRobotIndex_];
```

#### 2. **State Estimation**

The framework provides:
- Joint positions/velocities (from encoders)
- Body poses (from state estimator)
- Contact states (from force/torque sensors)

These are automatically updated by the framework before each control loop iteration.

#### 3. **Command Output**

The controller returns reference commands via:
- `CommandManager`: Aggregates all command terms
- `getValue()` method: Returns desired joint positions and velocities

The framework handles:
- PD control law application
- Torque limit enforcement
- Safety checks

---

## Observation System

### How Observations Are Computed

**Overview**: The observation system mimics the reinforcement learning training environment, computing the same features the neural network was trained on.

### Observation Pipeline

```
Robot State (joints, bodies)
    ↓
[ObservationManager evaluates all registered terms]
    ↓
Individual Observation Terms evaluate()
    ├─ MotionAnchorPosition: error between motion and robot anchor
    ├─ MotionAnchorOrientation: orientation error
    ├─ RobotBodyPosition: current body positions
    └─ RobotBodyOrientation: current body orientations
    ↓
Concatenated Observation Vector
    ↓
Fed to ONNX Policy
```

### Coordinate Frame Details

**Local Frame Transformations** (`MotionCommand.cpp:49-88`):

All observations are computed in the **robot's anchor body frame** (typically "pelvis"):

1. **Anchor Position** (line 49):
```cpp
// Transform motion anchor from world to robot anchor frame
anchorPoseReal.actInv(worldToInit_.act(anchorPos))
```

2. **Anchor Orientation** (line 57):
```cpp
// Rotation difference between motion and robot anchor
rot = anchorPoseReal.actInv(worldToInit_.act(anchorOri)).rotation()
// Convert to 6D representation (first 2 columns of rotation matrix)
```

3. **Body Positions** (line 66):
```cpp
// Each body position relative to robot anchor
for each body:
    bodyPoseLocal = anchorPoseReal.actInv(data.oMf[bodyIndices_[i]])
    position = bodyPoseLocal.translation()
```

4. **Body Orientations** (line 77):
```cpp
// Each body orientation relative to robot anchor
for each body:
    rot = anchorPoseReal.actInv(data.oMf[bodyIndices_[i]]).rotation()
    // Convert to 6D representation
```

### Why 6D Rotation Representation?

Instead of using quaternions (4D) or Euler angles (3D), the system uses **6D rotation representation** (first 2 columns of rotation matrix):
- **Continuous**: No discontinuities (unlike Euler angles)
- **Unique**: No ambiguity (unlike quaternions with q and -q)
- **ML-friendly**: Better for neural network training

---

## ONNX Model Integration

### Model Export Format

The ONNX models are exported from PyTorch training code with:
- **Inputs**:
  - `observations`: Float tensor [batch=1, obs_dim]
  - `time_step`: Float tensor [batch=1, 1]
- **Outputs**:
  - `joint_pos`: Joint positions [1, num_joints]
  - `joint_vel`: Joint velocities [1, num_joints]
  - `body_pos_w`: Body positions [num_bodies, 3]
  - `body_quat_w`: Body quaternions [num_bodies, 4] (wxyz)
- **Metadata**: Strings stored in ONNX metadata_props
  - `anchor_body_name`: "pelvis"
  - `body_names`: "left_hand,right_hand,pelvis"
  - `joint_names`: "joint1,joint2,..."
  - `kp`, `kd`: PD gains
  - Other training hyperparameters

See: https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/utils/exporter.py

### ONNX Runtime Execution

**Initialization** (`MotionOnnxPolicy.cpp` inherited from OnnxPolicy):
1. Load ONNX file
2. Parse metadata
3. Create input/output tensors
4. Build name-to-index mappings

**Inference Loop** (`MotionOnnxPolicy.cpp:17-41`):
```cpp
vector_t forward(const vector_t& observations) {
    // 1. Set time step input
    timeStep(0,0) = static_cast<float>(timeStep_++);
    inputTensors_[name2Index_.at("time_step")] = timeStep;

    // 2. Run base ONNX inference
    OnnxPolicy::forward(observations);

    // 3. Extract outputs
    jointPosition_ = outputTensors_["joint_pos"].row(0);
    jointVelocity_ = outputTensors_["joint_vel"].row(0);

    // 4. Parse body poses
    for each body:
        bodyPositions_.push_back(body_pos_w.row(i));
        bodyOrientations_.push_back(quaternion from body_quat_w.row(i));

    // 5. Return action
    return getLastAction();
}
```

**Key Insight**: The neural network outputs the entire motion trajectory (joint_pos, joint_vel, body poses) at the current timestep. The robot tracks these reference motions using its low-level PD controller.

---

## Data Flow

### Complete Control Loop

```
┌─────────────────────────────────────────────────────────────┐
│ 1. ROS 2 Control Manager calls update()                     │
└────────────┬────────────────────────────────────────────────┘
             │ 500 Hz (from controllers.yaml update_rate)
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. State Estimator Updates                                  │
│    - Joint positions/velocities (from encoders)             │
│    - Body poses (oMf[i]) (from IMU + kinematics)            │
└────────────┬────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ObservationManager Computes Observations                 │
│    - MotionAnchorPosition: error in anchor position         │
│    - MotionAnchorOrientation: error in anchor orientation   │
│    - RobotBodyPosition: current body positions              │
│    - RobotBodyOrientation: current body orientations        │
│    → Concatenated observation vector                        │
└────────────┬────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. ONNX Policy Inference                                    │
│    Input: [observations, time_step]                         │
│    Output: [joint_pos, joint_vel, body_pos_w, body_quat_w] │
└────────────┬────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. CommandManager Aggregates Commands                       │
│    - MotionCommandTerm.getValue()                           │
│    → Returns [joint_pos_ref, joint_vel_ref]                 │
└────────────┬────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. PD Controller (in framework)                             │
│    torque = kp*(q_ref - q) + kd*(v_ref - v)                │
└────────────┬────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Send to Robot Hardware/Simulation                        │
└─────────────────────────────────────────────────────────────┘
```

### Time Step Progression

```
t=0: reset() called
     ├─ ONNX policy: timeStep_ = startStep_ (e.g., 0)
     ├─ Align motion frame with robot
     └─ Run inference with dummy observations

t=1: First control loop
     ├─ Compute observations
     ├─ ONNX inference: timeStep_ = 0 → timeStep_++ = 1
     ├─ Get motion frame at t=0
     └─ Command robot to track

t=2: Second control loop
     ├─ Compute observations (tracking error)
     ├─ ONNX inference: timeStep_ = 1 → timeStep_++ = 2
     ├─ Get motion frame at t=1
     └─ Command robot to track

... (continues)
```

**Note**: The `start_step` parameter allows starting playback from a specific frame, useful for:
- Skipping initial frames
- Looping motions
- Debugging specific segments

---

## Configuration

### Launch Parameters (`real.launch.py`)

```yaml
robot_type: "g1"                    # Robot model
network_interface: "eth0"           # Network interface for Unitree SDK
policy_path: "/path/to/model.onnx" # ONNX model file
start_step: 0                       # Starting frame in motion
ext_pos_corr: false                 # External position correction (SLAM)
```

### Controller Configuration (`config/g1/controllers.yaml`)

```yaml
controller_manager:
  ros__parameters:
    update_rate: 500  # Control loop frequency (Hz)

walking_controller:  # MotionTrackingController
  ros__parameters:
    policy:
      path: "/path/to/model.onnx"
    motion:
      start_step: 0
    # Other parameters inherited from RlController...
```

### State Estimator Configuration

```yaml
state_estimator:
  ros__parameters:
    model:
      base_name: "pelvis"           # Base link for state estimation
      six_dof_contact_names:        # Contact bodies
        - "LL_FOOT"
        - "LR_FOOT"
    estimation:
      position:
        noise: 1e-2                 # Position estimate noise
        frame_id: "mid360_link"     # LiDAR frame for correction
```

---

## Coordinate Frames and Transformations

### Frame Hierarchy

```
world (fixed)
    ├─ pelvis (floating base)
    │   ├─ left_hip
    │   │   └─ left_knee
    │   │       └─ left_ankle
    │   │           └─ LL_FOOT
    │   ├─ right_hip
    │   │   └─ right_knee
    │   │       └─ right_ankle
    │   │           └─ LR_FOOT
    │   ├─ waist
    │   │   └─ chest
    │   │       ├─ left_shoulder
    │   │       │   └─ left_hand
    │   │       └─ right_shoulder
    │   │           └─ right_hand
```

### Key Transformations

#### 1. Motion Alignment (`MotionCommand.cpp:36-42`)

```
initToAnchor: Transform from motion's initial frame to anchor body
worldToAnchor: Transform from world to robot's anchor body
worldToInit: Alignment transform (only yaw orientation used)

worldToInit = worldToAnchor * initToAnchor^-1
```

#### 2. Observation Transformations (`MotionCommand.cpp:49-88`)

All observations are in the **robot's anchor frame**:

```
Motion anchor in world → Apply worldToInit → Transform to robot anchor frame
Robot bodies in world → Transform to robot anchor frame
```

#### 3. SE3 Operations

Using Pinocchio's SE3 class (Special Euclidean Group in 3D):
- `SE3.act(point)`: Transform point by SE3
- `SE3.actInv(point)`: Inverse transform
- `SE3.rotation()`: Get rotation matrix
- `SE3.translation()`: Get translation vector

---

## API Reference

### Valid APIs for Custom Development

If you want to extend or modify the controller, here are the key APIs:

#### From `legged_model` (Pinocchio wrapper)

```cpp
// Access robot model
const auto& pinModel = model_->getPinModel();
const auto& pinData = model_->getPinData();

// Get frame information
size_t frame_id = pinModel.getFrameId("frame_name");
pinocchio::SE3 frame_pose = pinData.oMf[frame_id];

// Get joint information
size_t num_joints = model_->getNumJoints();
vector_t joint_pos = pinData.q;  // Current joint positions
vector_t joint_vel = pinData.v;  // Current joint velocities
```

#### From `legged_rl_controllers`

```cpp
// CommandTerm interface
class CustomCommandTerm : public CommandTerm {
    size_t getSize() const override;  // Return command size
    vector_t getValue() override;      // Return command values
    void reset() override;             // Reset on activation
};

// ObservationTerm interface
class CustomObservation : public ObservationTerm {
    size_t getSize() const override;   // Return observation size
    vector_t evaluate() override;       // Compute observation
};

// Policy interface
class CustomPolicy : public Policy {
    void init() override;
    void reset() override;
    vector_t forward(const vector_t& obs) override;
};
```

#### From `controller_interface` (ROS 2)

```cpp
// Lifecycle callbacks
CallbackReturn on_init() override;
CallbackReturn on_configure(const State& prev) override;
CallbackReturn on_activate(const State& prev) override;
CallbackReturn on_deactivate(const State& prev) override;

// Parameter access
auto param = get_node()->get_parameter("param_name");
auto_declare("param_name", default_value);

// Logging
RCLCPP_INFO(get_node()->get_logger(), "message");
```

---

## Common Use Cases

### 1. Changing the ONNX Model

```bash
ros2 launch motion_tracking_controller real.launch.py \
    network_interface:=eth0 \
    policy_path:=/path/to/new_model.onnx
```

### 2. Starting from Different Frame

```bash
ros2 launch motion_tracking_controller real.launch.py \
    network_interface:=eth0 \
    policy_path:=model.onnx \
    start_step:=100  # Skip first 100 frames
```

### 3. Adding Custom Observations

1. Create new class inheriting from `ObservationTerm`
2. Implement `getSize()` and `evaluate()`
3. Register in `parserObservation()`

```cpp
class CustomObservation : public ObservationTerm {
    size_t getSize() const override { return 3; }
    vector_t evaluate() override {
        // Compute custom observation
        return vector_t::Zero(3);
    }
};

// In parserObservation():
if (name == "custom_obs") {
    observationManager_->addTerm(std::make_shared<CustomObservation>());
}
```

### 4. Adding Custom Commands

1. Create new class inheriting from `CommandTerm`
2. Implement `getSize()` and `getValue()`
3. Register in `parserCommand()`

---

## Debugging Tips

### 1. Visualizing Motion Trajectory

The ONNX policy outputs body positions/orientations - you can publish these as TF frames or markers for visualization in RViz.

### 2. Checking Observations

Add logging in `ObservationManager` to print observation values:
```cpp
RCLCPP_INFO_STREAM(logger_, "Observations: " << observations.transpose());
```

### 3. Frame Alignment Issues

Check the `worldToInit_` transform printed during `reset()`:
```cpp
std::cerr << worldToInit_ << std::endl;  // Line 46 in MotionCommand.cpp
```

### 4. ONNX Model Issues

Verify metadata parsing:
```bash
# Use Python to inspect ONNX metadata
python -c "
import onnx
model = onnx.load('model.onnx')
for prop in model.metadata_props:
    print(f'{prop.key}: {prop.value}')
"
```

---

## Additional Resources

- **legged_control2 Documentation**: https://qiayuanl.github.io/legged_control2_doc/
- **BeyondMimic Paper**: https://arxiv.org/abs/2508.08241
- **BeyondMimic Website**: https://beyondmimic.github.io/
- **Training Code**: https://github.com/HybridRobotics/whole_body_tracking
- **ONNX Exporter**: https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/utils/exporter.py

---

## Summary

The Motion Tracking Controller is a sophisticated system that:

1. **Loads** pre-trained ONNX models containing motion trajectories
2. **Aligns** the motion with the robot's current pose
3. **Tracks** reference motions using observations that match the RL training environment
4. **Commands** the robot via joint position/velocity references
5. **Leverages** the legged_control2 framework for robot abstraction and safety

The modular design allows easy extension with custom observations, commands, or policies while maintaining compatibility with the broader ROS 2 ecosystem.
