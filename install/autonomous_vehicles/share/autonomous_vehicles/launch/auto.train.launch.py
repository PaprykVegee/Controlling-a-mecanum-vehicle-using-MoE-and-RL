from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    # =====================
    # Paths
    # =====================
    world_path = "/home/developer/ros2_ws/src/new_models/mecanum.sdf"
    training_script_path = "/home/developer/ros2_ws/src/autonomous_vehicles/autonomous_vehicles/train_agent.py"
    working_dir = "/home/developer/ros2_ws/src/autonomous_vehicles"

    # =====================
    # Environment variables
    # =====================
    set_ros_domain_id = SetEnvironmentVariable(
        name='ROS_DOMAIN_ID',
        value='43'  # izolacja DDS (ważne!)
    )

    set_gz_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value='/home/developer/ros2_ws/src/models'
    )

    # =====================
    # Gazebo Sim
    # =====================
    gz_sim_world = ExecuteProcess(
        cmd=["gz", "sim", world_path, "-r"],
        output="screen"
    )

    # =====================
    # ROS <-> Gazebo bridge
    # =====================
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/world/mecanum_drive/model/vehicle_blue/link/camera_link/sensor/camera_sensor/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/world/mecanum_drive/model/vehicle_blue/link/lidar_2d_link/sensor/lidar_2d/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            # '/world/mecanum_drive/model/vehicle_blue/link/gps_link/sensor/gps/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat',
            "/world/mecanum_drive/model/vehicle_blue/link/lidar_link/sensor/lidar_3d/scan/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            #"/world/mecanum_drive/pose/info@geometry_msgs/msg/PoseArray[gz.msgs.Pose_V",
            "/model/vehicle_blue/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/model/vehicle_blue/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
        ],
        output='screen'
    )

    # =====================
    # Perception nodes
    # =====================
    image_node = Node(
        package='autonomous_vehicles',
        executable='image_node',
        name='image_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    lidar_node = Node(
        package='autonomous_vehicles',
        executable='lidar_node',
        name='lidar_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # =====================
    # Control node
    # =====================
    control_node = Node(
        package='autonomous_vehicles',
        executable='control_node',
        name='control_node',
        output='screen',
        respawn=True,
        arguments=['--ros-args', '--log-level', 'WARN'],
        parameters=[{'use_sim_time': True}]
    )

    # =====================
    # Offroad checker
    # =====================
    offroad_checker = Node(
        package='autonomous_vehicles',
        executable='offroad_checker',
        name='offroad_checker',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'centerline_file': '/home/developer/ros2_ws/src/xy.txt',
                'road_half_width': 2.0,
                'margin': 0.1,
                'stop_on_offroad': True,
                'odom_topic': '/model/vehicle_blue/odometry',
                'cmd_vel_topic': '/cmd_vel',
            }
        ]
    )

    # =====================
    # RL training (Python script)
    # =====================
    rl_training_node = ExecuteProcess(
        cmd=[
            'bash', '-c',
            f'source /opt/ros/humble/setup.bash && python3 {training_script_path}'
        ],
        cwd=working_dir,
        output='screen'
    )

    # =====================
    # Launch order (important!)
    # =====================
    return LaunchDescription([
        set_ros_domain_id,
        set_gz_resource_path,

        gz_sim_world,

        TimerAction(period=5.0, actions=[gz_bridge]),
        TimerAction(period=6.0, actions=[image_node, lidar_node]),
        TimerAction(period=7.0, actions=[control_node, offroad_checker]),
        TimerAction(period=9.0, actions=[rl_training_node]),
    ])
