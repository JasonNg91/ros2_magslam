# ros2_magslam
These packages can be used to perform mapping of the magnetic field using Gaussian Processes in ROS2 in psuedo real-time. Below you can find a description of various contents of this package.

## magslam package
### magmap.launch.py
Launches the magnetic field mapping node (magmap) with the parameters set in magslam/config/magmap.yaml. Magnetic field messages are expected as sensor_msgs/MagneticField, positions are obtained as a transform between two frames using TF2. These can be set in the config file.

The magmap node mode can be set using the magmap/mode topic of type std_msgs/String, possible modes are:
* Train: Create a map
* Test: Verify map against actual measurements
* None 

A log is written to maglog.csv in the current folder. Each row contains the following data:

<table>
      <tr><td colspan="8">Transform</td> <td colspan="7">Magnetic field</td> <td colspan="6">Magnetic field estimate (test only)</td></tr>
  <tr><td rowspan="2">Timestamp</td> <td colspan="3">Position</td> <td colspan="4">Rotation</td> <td rowspan="2">Timestamp</td> <td colspan="3">Robot</td> <td colspan="3">World</td> <td colspan="3">Mean</td> <td colspan="3">Covariance</td></tr>
      <tr><td>X</td><td>Y</td><td>Z</td><td>X</td><td>Y</td><td>Z</td><td>W</td><td>X</td><td>Y</td><td>Z</td><td>X</td><td>Y</td><td>Z</td><td>X</td><td>Y</td><td>Z</td><td>X</td><td>Y</td><td>Z</td>
</table>

### magview.launch.py
Launches the magnetic field visualisation node (magview) with the parameters set in magslam/config/magview.yaml. The sensor_frame and world_frame parameters should be set to the same values as in magslam/config/magmap.yaml.

While magview is running each frame is saved to mapframes/frame/[number].png relative to the current folder. Frames without the traveled path overlay are saved to mapframes/nopath/frame[number]\_[stamp].png. 

### csv_feed.launch.py
Launches the csv_feed node with the parameters set in magslam/config/csv_feed.yaml. This can be used to simulate magnetic field data comming in by reading it from a CSV file and publishing it, this allows magmap and magview to be testeds without a robot.

## magdrive package
### driver
The driver node can be used to command a ground robot to drive along a predetermined mapping path. The pose is obtained as a geometery_msgs/PoseStamped from Robot_1/pose. The drive commands are published as a geometry_msgs/Twist to cmd/vel. Three launch files with preset paths are included: 
* rectangle.launch.py
* rectangle_invert.launch.py
* grid.launch.py

## magslam_msgs package
This package contains the custom message types and services used by the magslam package.
