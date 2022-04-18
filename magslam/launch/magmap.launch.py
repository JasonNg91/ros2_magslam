import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config = os.path.join(get_package_share_directory('magslam'), 'config', 'magmap.yaml')

    return LaunchDescription([
      Node(
        package='magslam',
        executable='magmap',
        name='magmap',
        parameters=[config],
        output='screen',
        emulate_tty=True
      )
    ])
