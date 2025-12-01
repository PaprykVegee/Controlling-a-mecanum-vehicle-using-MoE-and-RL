from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    world_path = "/home/developer/ros2_ws/src/models/mecanum.sdf"
    gz_bridge_path = "/home/developer/ros2_ws/src/autonomous_vehicles/config/bridge_config.yaml"

    gz_sim_world = ExecuteProcess(
        cmd=["gz", "sim", world_path, "-r"],
        output="screen"
    )

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                   '/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image',
                   '/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                   '/world/mecanum_drive/model/vehicle_blue/link/gps_link/sensor/gps/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat'],
        output='screen'
    )


    control_node = Node(
        package='autonomous_vehicles',  
        executable='control_node',
        name='control_node',
        output="screen"
    )


    return LaunchDescription([
        gz_sim_world,
        gz_bridge,
        control_node
    ])
