import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import MagneticField

from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

import csv

class CsvFeed(Node):

	def __init__(self):
		super().__init__('csv_feed')
		
		# Load parameters
		self.declare_parameter('csv_path', '')
		self.csvPath = self.get_parameter('csv_path').get_parameter_value().string_value
		
		self.declare_parameter('interval', 0.05)
		self.interval = self.get_parameter('interval').value
		
		self.declare_parameter('train_ratio', 0.6)
		self.trainRatio = self.get_parameter('train_ratio').value
		
		self.declare_parameter('sensor_frame', 'imu_link')
		self.sensorFrame = self.get_parameter('sensor_frame').get_parameter_value().string_value
		
		self.declare_parameter('world_frame', 'odom')
		self.worldFrame = self.get_parameter('world_frame').get_parameter_value().string_value
		
		self.declare_parameter('mag_topic','imu/mag')
		self.magTopic = self.get_parameter('mag_topic').get_parameter_value().string_value

		# Setup publishers
		self.modePub = self.create_publisher(String, '/magmap/mode', 10)
		self.magPub = self.create_publisher(MagneticField, self.magTopic, 10)
		
		# Setup transform broadcaster
		self.tfBroadcaster = TransformBroadcaster(self)

		# Set timer at chosen interval
		self.timer = self.create_timer(self.interval, self.timer_callback)

		# Lists containing CSV data
		self.mag = []
		self.pos = []

		# Read data from CSV
		with open(self.csvPath, newline='') as csvFile:
			csvReader = csv.reader(csvFile, delimiter=',', quotechar='|')
			for row in csvReader:
				self.pos.append(list(map(float, row[0:3])))
				self.mag.append(list(map(float, row[3:6])))
		
		# Total size, size of training set, and size of test set
		self.size = len(self.mag)
		self.trainSize = int(self.size * self.trainRatio)
		self.testSize = self.size - self.trainSize
		
		# Current position in data set
		self.i = 0

	def timer_callback(self):
		# Determine mode by current index
		if (self.i < self.size):
			if (self.i < self.trainSize):
				self.mode = 'train'
			else:
				self.mode = 'test'
		else:
			self.mode = 'done'

		# Publish mode
		modeMsg = String()
		modeMsg.data = self.mode
		self.modePub.publish(modeMsg)
		
		# Do not continue if already done
		if self.mode == 'done':
			return

		# Publish magnetic field data
		magMsg = MagneticField()
		magMsg.magnetic_field.x = self.mag[self.i][0]
		magMsg.magnetic_field.y = self.mag[self.i][1]
		magMsg.magnetic_field.z = self.mag[self.i][2]
		magMsg.header.stamp = self.get_clock().now().to_msg()
		magMsg.header.frame_id = self.sensorFrame
		self.magPub.publish(magMsg)

		# Publish position data
		tf_msg = TransformStamped()
		tf_msg.header.frame_id = self.worldFrame
		tf_msg.header.stamp = self.get_clock().now().to_msg()
		tf_msg.child_frame_id = self.sensorFrame
		tf_msg.transform.translation.x = self.pos[self.i][0]
		tf_msg.transform.translation.y = self.pos[self.i][1]
		tf_msg.transform.translation.z = self.pos[self.i][2]
		self.tfBroadcaster.sendTransform(tf_msg)
		
		self.get_logger().info('Train: %i, Test: %i, Mode: %s, i: %i' % (self.trainSize, self.testSize, modeMsg.data, self.i))
		
		self.i = self.i + 1


def main(args=None):
	rclpy.init(args=args)

	csv_feed = CsvFeed()

	rclpy.spin(csv_feed)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	csv_feed.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
