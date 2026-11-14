cmake_minimum_required(VERSION 3.8)
project(motion_tracking_controller)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()


find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_library(SM_ONNXRUNTIME NAMES onnxruntime libonnxruntime.so.1 PATHS /opt/ros/$ENV{ROS_DISTRO}/lib REQUIRED)

ament_auto_add_library(
    ${PROJECT_NAME}
    SHARED
    DIRECTORY src
)

# prevent pluginlib from using boost
target_compile_definitions(${PROJECT_NAME} PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")
pluginlib_export_plugin_description_file(controller_interface motion_tracking_controller.xml)

target_include_directories(
    ${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen nlohmann_json::nlohmann_json ${SM_ONNXRUNTIME})

add_executable(sm_carry_policy_probe tools/sm_carry_policy_probe.cpp)
ament_target_dependencies(sm_carry_policy_probe legged_rl_controllers std_msgs)
target_include_directories(sm_carry_policy_probe PRIVATE include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(sm_carry_policy_probe ${PROJECT_NAME} ${SM_ONNXRUNTIME})
install(TARGETS sm_carry_policy_probe DESTINATION lib/${PROJECT_NAME})

ament_auto_package(
    INSTALL_TO_SHARE config launch tools
)
