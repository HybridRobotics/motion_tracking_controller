# Ubuntu 24.04 image based on ROS Jazzy (supports both x86_64 and arm64)
FROM ros:jazzy-perception-noble

ENV ROS_DISTRO=jazzy

# Disable interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    # APT / certificates / repository management / GPG keys
    # (apt-transport-https is usually unnecessary on Ubuntu 22.04)
    apt-get install -y --no-install-recommends \
      ca-certificates gnupg lsb-release software-properties-common && \
    \
    # Networking and troubleshooting tools
    apt-get install -y --no-install-recommends \
      iputils-ping iproute2 wget curl net-tools && \
    \
    # Build / debug / code management tools
    apt-get install -y --no-install-recommends \
      build-essential cmake gdb git git-lfs vim python3-vcstool && \
    \
    # Common C++ math and utility libraries
    apt-get install -y --no-install-recommends \
      libboost-all-dev libeigen3-dev libyaml-cpp-dev && \
    \
    # JSON processing / code formatting / miscellaneous utilities
    apt-get install -y --no-install-recommends \
      jq clang-format unzip ncdu && \
    \
    # GUI support (kept since GUI is required; gnome-terminal is large but convenient)
    apt-get install -y --no-install-recommends \
      dbus-x11 gnome-terminal && \
    \
    # Hardware / USB / joystick support
    apt-get install -y --no-install-recommends \
      udev usbutils joystick && \
    \
    # Python toolchain (pip)
    apt-get install -y --no-install-recommends \
      python3-pip && \
    \
    git lfs install

    
    # ROS dependencies
RUN apt-get install -y --no-install-recommends \
        ros-jazzy-plotjuggler-ros ros-jazzy-pinocchio

    # Install additional dependencies
    # unitree_sdk2
RUN apt-get install -y --no-install-recommends \
        libspdlog-dev libfmt-dev && \
    \
    # unitree_ros2
    apt-get install -y --no-install-recommends \
        ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-rosidl-generator-dds-idl && \
    \
    # unitree_mujoco
    apt-get install -y --no-install-recommends \
        libglfw3-dev

    # Add apt source
RUN echo "deb [trusted=yes] https://github.com/qiayuanl/legged_buildfarm/raw/noble-jazzy-amd64/ ./" | sudo tee /etc/apt/sources.list.d/qiayuanl_legged_buildfarm.list && \
    echo "yaml https://github.com/qiayuanl/legged_buildfarm/raw/noble-jazzy-amd64/local.yaml jazzy" | sudo tee /etc/ros/rosdep/sources.list.d/1-qiayuanl_legged_buildfarm.list  && \
    echo "deb [trusted=yes] https://github.com/qiayuanl/unitree_buildfarm/raw/noble-jazzy-amd64/ ./" | sudo tee /etc/apt/sources.list.d/qiayuanl_unitree_buildfarm.list && \
    echo "yaml https://github.com/qiayuanl/unitree_buildfarm/raw/noble-jazzy-amd64/local.yaml jazzy" | sudo tee /etc/ros/rosdep/sources.list.d/1-qiayuanl_unitree_buildfarm.list && \
    echo "deb [trusted=yes] https://github.com/qiayuanl/simulation_buildfarm/raw/noble-jazzy-amd64/ ./" | sudo tee /etc/apt/sources.list.d/qiayuanl_simulation_buildfarm.list && \
    echo "yaml https://github.com/qiayuanl/simulation_buildfarm/raw/noble-jazzy-amd64/local.yaml jazzy" | sudo tee /etc/ros/rosdep/sources.list.d/1-qiayuanl_simulation_buildfarm.list && \
    sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        ros-jazzy-legged-control-base ros-jazzy-mujoco-ros2-control ros-jazzy-unitree-description ros-jazzy-unitree-systems

# Copy workspace setup script into the container
COPY workspace-config.sh /workspace-config.sh

# Use workspace-config.sh as the container entrypoint
ENTRYPOINT ["/workspace-config.sh"]
