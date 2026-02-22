#!/bin/bash

# Default build type is Release
BUILD_TYPE="${1:-Release}"

echo "Building with CMAKE_BUILD_TYPE=$BUILD_TYPE"

# Get project root directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"/../../../
echo "Project directory: $PROJECT_DIR"
mkdir -p $PROJECT_DIR/lib
mkdir -p $PROJECT_DIR/src

# Clone repos
cd $PROJECT_DIR/src
git clone https://github.com/qiayuanl/unitree_bringup.git

# Build
cd $PROJECT_DIR
rosdep install --from-paths src --ignore-src -r -y

source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
    --packages-up-to unitree_bringup motion_tracking_controller \
    --cmake-args \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE 