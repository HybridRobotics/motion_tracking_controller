import os

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    DeclareLaunchArgument,
    OpaqueFunction,
    SetLaunchConfiguration,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from legged_bringup.launch_utils import get_controller_names, generate_temp_config, download_wandb_onnx, control_spawner


def setup_controllers(context):
    robot_type_value = LaunchConfiguration('robot_type').perform(context)
    policy_path_value = LaunchConfiguration('policy_path').perform(context)
    wandb_path_value = LaunchConfiguration('wandb_path').perform(context)
    start_step_value = LaunchConfiguration('start_step').perform(context)
    ext_pos_corr = LaunchConfiguration('ext_pos_corr').perform(context)

    if not policy_path_value and wandb_path_value:
        policy_path_value = download_wandb_onnx(wandb_path_value)

    kv_pairs = []
    if policy_path_value:
        abs_path = os.path.abspath(os.path.expanduser(os.path.expandvars(policy_path_value)))
        kv_pairs.append(('walking_controller.policy.path', abs_path))
    if start_step_value:
        kv_pairs.append(('walking_controller.motion.start_step', start_step_value))
    if ext_pos_corr.lower() in ["true", "1", "yes"]:
        kv_pairs.append(('state_estimator.estimation.contact.height_sensor_noise', 1e10))
        kv_pairs.append(('state_estimator.estimation.position.topic', "/glim/odom"))

    controllers_config_path = f'config/{robot_type_value}/controllers.yaml'
    temp_controllers_config_path = generate_temp_config(
        controllers_config_path,
        'motion_tracking_controller',
        kv_pairs
    )

    set_controllers_yaml = SetLaunchConfiguration(
        name='controllers_yaml',
        value=temp_controllers_config_path
    )

    all_controllers = get_controller_names(controllers_config_path, 'motion_tracking_controller')
    active_list = ["state_estimator", "standby_controller"]
    inactive_list = [c for c in all_controllers if c not in active_list]

    param_file = LaunchConfiguration('controllers_yaml')
    active_spawner = control_spawner(active_list, param_file=param_file)
    inactive_spawner = control_spawner(inactive_list, inactive=True, param_file=param_file)

    return [set_controllers_yaml, active_spawner, inactive_spawner]


def generate_launch_description():
    robot_type = LaunchConfiguration('robot_type')
    network_interface = LaunchConfiguration('network_interface')
    urdf_name = PythonExpression(["'g1' if '", robot_type, "' == 'g1' else 'sdk1'"])

    robot_description_command = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("unitree_description"),
            "urdf",
            urdf_name,
            "robot.xacro"
        ]),
        " ", "robot_type:=", robot_type,
        " ", "simulation:=", "false",
        " ", "network_interface:=", network_interface
    ])

    robot_description = {"robot_description": robot_description_command}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {
            'publish_frequency': 500.0,
        }],
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, LaunchConfiguration('controllers_yaml')],
        output="both",
        respawn=True,
    )

    controllers_opaque_func = OpaqueFunction(function=setup_controllers)

    # Exclude all Unitree topics... it should start from the same namespace, fuck Unitree!
    exclude_regex = (
        r'(/EstimatorData|/SymState(_back)?|/api/.*'
        r'|/arm/action/state|/arm_sdk'
        r'|/audio_msg|/audiosender|/config_change_status'
        r'|/dex3/(left|right)/(cmd|state)'
        r'|/frontvideostream|/gnss'
        r'|/gpt_(cmd|state)|/gptflowfeedback'
        r'|/lf/(bmsstate|dex3/(left|right)/state|lowstate|mainboardstate|'
        r'odommodestate|secondary_imu|sportmodestate)'
        r'|/low(cmd|state)|/multiplestate|/odommodestate'
        r'|/parameter_events|/public_network_status|/rosout'
        r'|/rtc/(state|status)|/secondary_imu|/selftest'
        r'|/servicestate(activate)?|/slam_info|/sportmodestate'
        r'|/utlidar/range_info|/videohub/inner'
        r'|/webrtc(req|res)|/wirelesscontroller)'
        r'|/controller_manager/introspection_data/full'
        r'|/controller_manager/statistics/full'
    )

    rosbag2 = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record', '-s', 'mcap', '-a',  # record all topics
            '--exclude-regex', exclude_regex,  # skip those that match the regex
        ],
        output='screen',
    )

    teleop = PathJoinSubstitution([
        FindPackageShare('unitree_bringup'),
        'launch',
        'teleop.launch.py'
    ])

    return LaunchDescription([
        DeclareLaunchArgument('robot_type', default_value='g1'),
        DeclareLaunchArgument('network_interface'),
        DeclareLaunchArgument(
            'policy_path',
            default_value='',
            description='Absolute or ~-expanded path for walking_controller.policy.path'
        ),
        DeclareLaunchArgument(
            'start_step',
            default_value='0',
            description='Integer start step for walking_controller.motion.start_step'
        ),
        DeclareLaunchArgument(
            'ext_pos_corr',
            default_value='false',
            description='Enable external position correction'
        ),
        DeclareLaunchArgument(
            'wandb_path',
            default_value='',
            description='W&B run path to download ONNX from (used when policy_path is empty)'
        ),
        controllers_opaque_func,
        control_node,
        node_robot_state_publisher,
        rosbag2,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(teleop),
            launch_arguments={'robot_type': robot_type}.items()
        )
    ])
