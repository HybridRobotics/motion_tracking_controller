"""SuperMimic carry policy in MuJoCo only. No real-hardware backend is launched."""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    assets = Path(LaunchConfiguration('assets').perform(context)).expanduser().resolve()
    control = json.loads((assets/'control.json').read_text())
    own = Path(get_package_share_directory('motion_tracking_controller'))
    description = Path(get_package_share_directory('unitree_description'))
    robot = {'robot_description': Command(['xacro ',str(description/'urdf/g1/robot.xacro'),
                                          ' robot_type:=g1 simulation:=mujoco'])}
    parameters = str(own/'config/g1/sm_carry_controllers.yaml')
    overrides = {'walking_controller': {'ros__parameters': {
        'policy.path':str(assets/'actor.onnx'),
        'sm_carry.control_path':str(assets/'control.json'),
        'sm_carry.reference_path':str(assets/'reference.json'),
        'sm_carry.align_heading':True,
        'sm_carry.debug':True}}}
    # Keep generated configuration in the caller's asset directory, not a shared /tmp filename.
    import yaml
    merged = yaml.safe_load(Path(parameters).read_text())
    merged['walking_controller']['ros__parameters'].update(overrides['walking_controller']['ros__parameters'])
    merged['standby_controller']['ros__parameters'].update({
        'joint_names':control['joint_names'], 'default_position':control['default_joint_position_rad'],
        'kp':control['kp'],'kd':control['kd']})
    generated = assets/'controllers.generated.yaml'
    generated.write_text(yaml.safe_dump(merged))
    # A local XML copy keeps the installed robot description untouched. A fixed
    # trackcom camera follows the simulated robot without mouse-based selection.
    tree = ET.parse(description/'mjcf/g1.xml')
    compiler = tree.getroot().find('compiler')
    compiler.set('meshdir', str((description/'mjcf'/compiler.get('meshdir', '')).resolve()))
    world = tree.getroot().find('worldbody')
    ET.SubElement(world, 'camera', name='sm_carry_follow', mode='trackcom',
                  pos='1.8 -2.4 1.5', xyaxes='0.8 0.6 0 -0.12 0.16 0.98')
    armatures = dict(zip(control['joint_names'],control['armature']))
    for joint in world.iter('joint'):
        if joint.get('name') in armatures:
            joint.set('armature', str(armatures[joint.get('name')]))
    model_xml = assets/'sim.generated.xml'
    tree.write(model_xml)
    model_file = '/'+os.path.relpath(model_xml,description)
    simulator = Node(package='mujoco_sim_ros2',executable='mujoco_sim',output='screen',
        parameters=[{'model_package':'unitree_description','model_file':model_file,
                     'physics_plugins':['mujoco_ros2_control::MujocoRos2ControlPlugin'], 'use_sim_time':True},
                    robot,str(generated),str(assets/'startup.yaml')],
        on_exit=Shutdown(reason='SM carry MuJoCo simulation exited'))
    publisher = Node(package='robot_state_publisher',executable='robot_state_publisher',
                     parameters=[robot,{'use_sim_time':True}],output='screen')
    spawner = Node(package='controller_manager',executable='spawner',output='screen',
                   arguments=['state_estimator','walking_controller','--param-file',str(generated)])
    return [simulator,publisher,TimerAction(period=float(LaunchConfiguration('controller_start_delay').perform(context)),actions=[spawner])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('assets',description='Prepared SM carry asset directory'),
        DeclareLaunchArgument('controller_start_delay',default_value='0.0'),
        OpaqueFunction(function=setup)])
