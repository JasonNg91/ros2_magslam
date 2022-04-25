# ros2_magslam
These packages can be used to perform mapping of the magnetic field using Gaussian Processes in ROS2 in psuedo real-time. Below you can find a description of various contents of this package.

## magslam package
### magmap.launch.py
Launches the magnetic field mapping node (magmap) with the parameters set in magslam/config/magmap.yaml. Magnetic field messages are expected as sensor_msgs/MagneticField, positions are obtained as a transform between two frames using TF2. These can be set in the config file.

### magview.launch.py
Lunaches the magnetic field visualisation node (magview) with the parameters set in magslam/config/magview.yaml. The sensor_frame and world_frame parameters should be set to the same values as in magslam/config/magmap.yaml.

### csv_feed.launch.py
Launches thee csv_feed node with the parameters set in magslam/config/csv_feed.yaml. This can be used to simulate magnetic field data comming in by reading it from a CSV file and publishing it, this allows magmap and magview to be testeds without a robot.

## magdrive pacakage
### driver
The driver node can be used to command a ground robot to drive along a predetermined mapping path. The pose is obtained as a geometery_msgs/PoseStamped from Robot_1/pose. The drive commands are published as a geometry_msgs/Twist to cmd/vel. This node has no config file (yet), configuration can be done in the sourcecode in magdrive/src/driver.cpp

## magslam_msgs package
This package contains the custom message types and services used by the magslam package.
