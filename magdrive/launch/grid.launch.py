from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(get_package_share_directory('magdrive'), 'config', 'driver.yaml')

    return LaunchDescription([
        Node(
            package="magdrive",
            executable="driver",
            output="screen",
            emulate_tty=True,
            parameters=[
                config,
                {
                "path_mode": "grid",
                "x_min": -1.5,
                "y_min": -1.5,
                "x_max": 1.5,
                "y_max": 1.5,
                "density:": 0.5
                }
            ]
        )
    ])