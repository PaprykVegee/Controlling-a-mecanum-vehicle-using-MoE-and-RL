from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    world_path = "/home/developer/ros2_ws/src/models/mecanum.sdf"

    set_gz_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value='/home/developer/ros2_ws/src/models'
    )

    gz_sim_world = ExecuteProcess(
        cmd=["gz", "sim", world_path, "-r"],
        output="screen"
    )

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/world/mecanum_drive/model/vehicle_blue/link/lidar_2d_link/sensor/lidar_2d/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/world/mecanum_drive/model/vehicle_blue/link/gps_link/sensor/gps/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat',
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar_3d/scan/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/model/vehicle_blue/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
        ],
        output='screen'
    )

    image_node = Node(
        package='autonomous_vehicles',  # tu wpisz swój package z SegmentationNode
        executable='image_node',  # nazwa executable z setup.py / entry_point
        name='image_node',
        output='screen'
    )

    lidar_node = Node(
        package='autonomous_vehicles',  # tu wpisz swój package z SegmentationNode
        executable='lidar_node',  # nazwa executable z setup.py / entry_point
        name='lidar_node',
        output='screen'
    )

    lidar_2d_node = Node(
        package='autonomous_vehicles',
        executable='lidar_2d_node',
        name='lidar_2d_node',
        output='screen'
    )


    gps_node = Node(
        package='autonomous_vehicles',
        executable='gps_node',
        name='gps_node',
        output='screen'
    )


    control_node = Node(
        package='autonomous_vehicles',
        executable='control_node',
        name='control_node',
        output="screen",
        arguments=['--ros-args', '--log-level', 'WARN'],
        
    )
    offroad_checker = Node(
        package="autonomous_vehicles",
        executable="offroad_checker",
        name="offroad_checker",
        output="screen",
        parameters=[{
            "centerline_file": "/home/developer/ros2_ws/src/xy_gt.txt",
            "road_half_width": 2.0,   # połowa szerokości jezdni (ustaw pod swój tor)
            "margin": 0.1,
            "stop_on_offroad": False,
            "odom_topic": "/model/vehicle_blue/odometry",
            "cmd_vel_topic": "/cmd_vel",
        }]
    )

    return LaunchDescription([
        set_gz_resource_path,
        gz_sim_world,
        gz_bridge,
        image_node,
        lidar_node,
        lidar_2d_node,
        gps_node,
        control_node,
        offroad_checker
    ])
