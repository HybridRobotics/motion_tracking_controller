# BeyondMimic Motion Tracking Inference

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

This repository provides the inference pipeline for motion tracking policies in BeyondMimic. The pipeline is implemented
in C++ using the ONNX CPU inference engine. Model parameters (joint order, impedance, etc.)
are stored in ONNX metadata, and the reference motion is returned via the `forward()` function.
See [this script](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/utils/exporter.py)
for details on exporting models.

This repo also serves as an example of how to implement a custom controller using the
[legged_control2](https://qiayuanl.github.io/legged_control2_doc/) framework.

## Install and Build (docker)
The Dockerfile and helper scripts are provided to simplify installation and build.
Make sure [Docker](https://docs.docker.com/engine/install/ubuntu/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) are installed first.
```bash
# Host
git clone https://github.com/HybridRobotics/motion_tracking_controller.git
cd motion_tracking_controller/docker
docker compose up -d --build
docker exec -it wbt_ws bash
```

```bash
# container
cd /wbt_ws/src/motion_tracking_controller
./scripts/colcon-config.sh Release
```
This script automatically clones and builds the required dependencies.
See [Basic Usage](#basic-usage) for usage instructions.

## Installation

### Dependencies

This software is built on
the [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html#ubuntu-deb-packages), which
needs to be installed first. Additionally, this code base depends on `legged_control2`.

### Install `legged_control2`

Pre-built binaries for `legged_control2` are available on ROS 2 Jazzy. We recommend first reading
the [full documentation](https://qiayuanl.github.io/legged_control2_doc/overview.html).

Specifically, For this repo, follow
the [Debian Source installation](https://qiayuanl.github.io/legged_control2_doc/installation.html#debian-source-recommended).
Additionally, install Unitree-specific packages:

```bash
# Add debian source
echo "deb [trusted=yes] https://github.com/qiayuanl/unitree_buildfarm/raw/noble-jazzy-amd64/ ./" | sudo tee /etc/apt/sources.list.d/qiayuanl_unitree_buildfarm.list
echo "yaml https://github.com/qiayuanl/unitree_buildfarm/raw/noble-jazzy-amd64/local.yaml jazzy" | sudo tee /etc/ros/rosdep/sources.list.d/1-qiayuanl_unitree_buildfarm.list
sudo apt-get update
```

```bash
# Install packages
sudo apt-get install ros-jazzy-unitree-description
sudo apt-get install ros-jazzy-unitree-systems
```

### Build Package

After installing `legged_control2`, you can build this package. You’ll also need the
`unitree_bringup` repo, which contains utilities not included in the pre-built binaries.

Create a ROS 2 workspace if you don't have one. Below we use `~/colcon_ws` as an example.

```bash
mkdir -p ~/colcon_ws/src
```

Clone two repo into the `src` of workspace.

```bash
cd ~/colcon_ws/src
git clone https://github.com/qiayuanl/unitree_bringup.git
git clone https://github.com/HybridRobotics/motion_tracking_controller.git
cd ../
```

Install dependencies automatically:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

Build the packages:

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelwithDebInfo --packages-up-to unitree_bringup
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelwithDebInfo --packages-up-to motion_tracking_controller
source install/setup.bash
```

## Basic Usage

### Sim-to-Sim

We provide a launch file for running the policy in MuJoCo simulation.

```bash
# Load policy from WandB
ros2 launch motion_tracking_controller mujoco.launch.py wandb_path:=<your_wandb_run_path>
```

```bash
# OR load policy from local ONNX file (should be absolute or start with `~`)
ros2 launch motion_tracking_controller mujoco.launch.py policy_path:=<your_onnx_file_path>
```

### Real Experiments

> ⚠️ **Disclaimer**  
> Running these models on real robots is **dangerous** and entirely at your own risk.  
> They are provided **for research only**, and we accept **no responsibility** for any harm, damage, or malfunction.

1. Connect to the robot via ethernet cable.
2. Set the ethernet adapter to static IP: `192.168.123.11`.
3. Use `ifconfig` to find the `<network_interface>`, (e.g.,`eth0` or `enp3s0`).

```bash
# Load policy from WandB
ros2 launch motion_tracking_controller real.launch.py network_interface:=<network_interface> wandb_path:=<your_wandb_run_path>
```

```bash
# OR load policy from local ONNX file (should be absolute or start with `~`)
ros2 launch motion_tracking_controller real.launch.py network_interface:=<network_interface> policy_path:=<your_onnx_file_name>.onnx
```

The robot should enter standby controller in the beginning.
Use the Unitree remote (joystick) to start and stop the policy:

- Standby controller (joint position control): `L1 + A`
- Motion tracking controller (the policy): `R1 + A`
- E-stop (damping): `B`

## Code Structure

This section will be especially helpful if you decide to write your own legged_control2 controller.
For a minimal starting point, check
the [legged_template_controller](https://github.com/qiayuanl/legged_template_controller).

Below is an overview of the code structure for this repository:

- **`include`** or **`src`**
    - **`MotionTrackingController`** Manages observations (like an RL environment) and passes them to the policy.

    - **`MotionOnnxPolicy`** Wraps the neural network, runs inference, and extracts reference motion from the ONNX file.

    - **`MotionCommand`** Defines observation terms aligned with the training code.


- **`launch`**
    - Includes launch files like `mujoco.launch.py` and `real.launch.py` for simulation and real robot execution.
- **`config`**
    - Stores configuration files for standby controller and state estimation params.
