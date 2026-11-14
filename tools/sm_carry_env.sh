# Source from the NUC helper scripts. Override these paths for another machine.
unset LD_LIBRARY_PATH PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH ROS_ROOT ROS_VERSION ROS_DISTRO
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PYTHONNOUSERSITE=1
source /opt/ros/jazzy/setup.bash
source "${SM_CARRY_UNDERLAY:-$HOME/colcon_ws}/install/setup.bash"
source "${SM_CARRY_WORKSPACE:-$HOME/sm_carry_box_ws}/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-72}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
