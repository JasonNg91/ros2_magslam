from cmath import log
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import MagneticField
from std_msgs.msg import String
from magslam_msgs.msg import MagGPMem

from magslam_msgs.srv import GetMag

from tf2_ros import TransformException
from tf2_ros import TransformListener
from tf2_ros.buffer import Buffer

from .gp.magfield import GPMagMap
from .gp.DGP.plotfunctions import plotTimeSeriesWithUncertainty

import numpy as np
import os
import matplotlib.pyplot as plt
import time

from multiprocessing import shared_memory

import csv

# ROS time to float
def timeToFloat(t):
	return t.sec + t.nanosec/1000000000

# Quaternion to rotation matrix
def rotationMatrix(q):
	return [[1 - 2*q.y**2 - 2*q.z**2, 2*q.x*q.y - 2*q.w*q.z,   2*q.x*q.z + 2*q.w*q.y],
		[2*q.x*q.y + 2*q.w*q.z,   1 - 2*q.x**2 - 2*q.z**2, 2*q.y*q.z - 2*q.w*q.x],
		[2*q.x*q.z - 2*q.w*q.y,   2*q.y*q.z + 2*q.w*q.x,   1 - 2*q.x**2 - 2*q.y**2]]


def hamprod(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def rotateVector(vector, R):
	p = [0, vector[0], vector[1], vector[2]]

	r1 = [R.w, R.x, R.y, R.z]
	r2 = [R.w, -R.x, -R.y, -R.z]

	pr = hamprod(hamprod(r1, p), r2)

	return [pr[1], pr[2], pr[3]]

class Magmap(Node):

	def __init__(self):
		super().__init__('magmap')

		# Parameter setup
		self.declare_parameter('sensor_frame', 'imu_link')
		self.sensorFrame = self.get_parameter('sensor_frame').get_parameter_value().string_value
		self.get_logger().info('sensor_frame: ' + self.sensorFrame)
		
		self.declare_parameter('world_frame', 'odom')
		self.worldFrame = self.get_parameter('world_frame').get_parameter_value().string_value
		self.get_logger().info('world_frame: ' + self.worldFrame)
		
		self.declare_parameter('mag_topic','imu/mag')
		self.magTopic = self.get_parameter('mag_topic').get_parameter_value().string_value
		self.get_logger().info('mag_topic: ' + self.magTopic)
		
		self.declare_parameter('boundaries',[2.5, 2.5, 1.0])
		self.boundaries = self.get_parameter('boundaries').value
		self.get_logger().info('boundaries: ' + str(self.boundaries))
		
		# The magmap object handles training and estimations
		self.magmap = GPMagMap(self.boundaries)

		# Mode subscription, valid modes are 'train', 'test', and 'none'
		self.mode_sub = self.create_subscription(
			String,
			'/magmap/mode',
			self.modeCallback,
			10)
		
		# Magnetic field subscription
		self.mag_sub = self.create_subscription(
			MagneticField,
			self.magTopic,
			self.magCallback,
			10)

		# Transform listener for sensor position and orientation in world
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		self.tf_timer = self.create_timer(1/100, self.getTF)		#Timer to check for new transform 100 times per second
		
		# GP Publishing
		self.GPPub = self.create_publisher(MagGPMem, '/magmap/gp', 10)
		self.GPPubTtimer = self.create_timer(1/4, self.pubGPMem)		#Timer to publish GP four times a second
		self.SigmaMem = shared_memory.SharedMemory(create=True, size=self.magmap.Sigma.nbytes)
		self.muMem = shared_memory.SharedMemory(create=True, size=self.magmap.mu.nbytes)
		self.sharedSigma = np.ndarray(self.magmap.Sigma.shape, dtype=self.magmap.Sigma.dtype, buffer=self.SigmaMem.buf)
		self.sharedMu = np.ndarray(self.magmap.mu.shape, dtype=self.magmap.mu.dtype, buffer=self.muMem.buf)

		# Service for requesting magneticfield estimates
		self.getMagSrv = self.create_service(GetMag, 'get_mag', self.getMagCallback)

		# Magnetic fields and transforms for which no match has been found yet since the last one
		self.magBuffer = []
		self.transBuffer = []

		# Timestamp of previous transform, used to avoid duplicates
		self.prevTransStamp = 0

		# Current mode
		self.mode = 'none'
		
		# Lists of estimations and actual fields during testing
		self.means = []
		self.covs = []
		self.actual = []
		
		# Variables for statistics output
		self.trainSize = 0
		self.testSize = 0
		self.prevStamp = 0
		self.calcTime = 0
		self.GPPubTime = 0

		# Setup CSV logging
		self.log_file = open('maglog.csv','w')
		self.log_writer = csv.writer(self.log_file)

		self.get_logger().info('Magmap node initiated.')
		
	# Called when a new message is received on the mode topic
	def modeCallback(self, string_msg):
		# Start of test
		if (self.mode != 'test' and string_msg.data == 'test'):
			# Clear previous test results
			self.means = []
			self.actual = []
			self.testSize = 0
		
		# End of test
		if (self.mode == 'test' and string_msg.data != 'test'):
			# Convert lists to numpy arrays
			actualArray = np.array(self.actual)
			meanArray = np.array(self.means)
			covArray = np.array(self.covs)

			np.savetxt("Actual.csv", actualArray, delimiter=",")
			np.savetxt("Mean.csv", meanArray, delimiter=",")
			np.savetxt("Cov.csv", covArray, delimiter=",")
			
			# Plot test results
			plotTimeSeriesWithUncertainty([0, actualArray],(meanArray, covArray))
			plt.show()
		
		# Set the mode to the new recieved mode
		self.mode = string_msg.data
	
	# Called when a new message is received on the magnetic field topic
	def magCallback(self, mag_msg):
		self.magBuffer.append(mag_msg)	# Add to lists of magnetic fields
		self.findMatch()			# Look for matches of magnetic fields and positions by timestamp

	# Called at a set interval to get the IMU's poses
	def getTF(self):
		# Try to get transform from world frame to IMU, display warning if it fails.
		try:
			now = rclpy.time.Time()
			trans = self.tf_buffer.lookup_transform(self.worldFrame, self.sensorFrame, now)
		except TransformException as ex:
			self.get_logger().warn(f'Could not transform {self.sensorFrame} to {self.worldFrame}: {ex}')
			return
		
		# Timestamp at which the transform was last updated
		stamp = timeToFloat(trans.header.stamp)

		# Check if time is flowing in the right direction
		if (stamp > self.prevTransStamp):
			self.prevTransStamp = stamp		# Save this timestamp to check the flow of time next time
			self.transBuffer.append(trans)	# Add the transform to the list of transforms
			self.findMatch()			# Look for matches of magnetic fields and positions by timestamp
	
	# Functon to match magnetic fields and positions to each other for the smallest timestamp difference
	def findMatch(self):
		newMagBuffer = self.magBuffer
		newTransBuffer = self.transBuffer

		for trans_i in range(len(self.transBuffer)):
			transStamp = timeToFloat(self.transBuffer[trans_i].header.stamp)

			for mag_i in range(len(self.magBuffer) - 1):
				magStampA = timeToFloat(self.magBuffer[mag_i].header.stamp)
				magStampB = timeToFloat(self.magBuffer[mag_i+1].header.stamp)

				if magStampA < transStamp and magStampB > transStamp:
					dStampA = transStamp - magStampA
					dStampB = magStampB - transStamp

					if dStampA < dStampB:
						magr, magw, pos = self.addPair(self.magBuffer[mag_i], self.transBuffer[trans_i])
						newMagBuffer = self.magBuffer[mag_i+1:]
						self.printStatus(dStampA, transStamp, magr, magw, pos)
					else:
						magr, magw, pos = self.addPair(self.magBuffer[mag_i+1], self.transBuffer[trans_i])
						newMagBuffer = self.magBuffer[mag_i+2:]
						self.printStatus(dStampB, transStamp, magr, magw, pos)

					newTransBuffer = self.transBuffer[trans_i+1:]

		self.transBuffer = newTransBuffer
		self.magBuffer = newMagBuffer

	# Function to get data to and from the magmap object
	def addPair(self, mag_msg, tf_msg):
		# Put position and magnetic field in a list
		magR = [mag_msg.magnetic_field.x, mag_msg.magnetic_field.y, mag_msg.magnetic_field.z]
		pos = [tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z]

		#Rotate magnetic field in world frame
		#rotMat = rotationMatrix(tf_msg.transform.rotation)
		#magW = np.dot(magR, rotMat)

		magW = rotateVector(magR, tf_msg.transform.rotation)

		log_row = [self.mode, 
					timeToFloat(tf_msg.header.stamp), tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z,
					tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w,
					timeToFloat(mag_msg.header.stamp), magR[0], magR[1], magR[2], magW[0], magW[1], magW[2]]
		
		calcStart = time.time()

		if self.mode == 'train':				# If in train mode
			self.magmap.train(np.array(pos),np.array(magW))	# Add the data to the map
			self.trainSize = self.trainSize + 1
		elif self.mode == 'test':				# If in test mode
			mean, cov = self.magmap.predict(np.array(pos))	# Get predictions for the position
			self.means.append(mean)				# Prediciton mean
			self.covs.append(cov)					# Prediciton covariance
			self.actual.append(magW)				# Actual field
			self.testSize = self.testSize + 1
			log_row.extend(mean)
			log_row.extend(cov)

		self.calcTime = time.time() - calcStart

		self.log_writer.writerow(log_row)

		return magR, magW, pos
	
	# Output stats to terminal
	def printStatus(self, dStamp, stamp, magR, magW, pos):
		os.system('clear')
		print('Mode: ' + self.mode)
		print('Buffer sizes: ')
		print(' - Transform: ' + str(len(self.transBuffer)))
		print(' - Magfield:  ' + str(len(self.magBuffer)))
		print('Timing:')
		print(' - dt:    ' + str(dStamp))
		print(' - T:     ' + str(stamp-self.prevStamp))
		print(' - calc:  ' + str(self.calcTime))
		print(' - GPPub: ' + str(self.GPPubTime))
		print('Data:')
		print(' - Magfied: ')
		print('   - World: ' + str(magW))
		print('   - Robot: ' + str(magR))
		print(' - Position:' + str(pos))
		print('Frames:')
		print(' - sensor: ' + self.sensorFrame)
		print(' - world:  ' + self.worldFrame)
		print('Train points: ' + str(self.trainSize))
		print('Test points:  ' + str(self.testSize))
		
		self.prevStamp = stamp

	def pubGPMem(self):
		GPPubStart = time.time()
		
		self.sharedMu[:] = self.magmap.mu[:]
		self.sharedSigma[:] = self.magmap.Sigma[:]

		GPMsg = MagGPMem()
		
		GPMsg.sigma_name = self.SigmaMem.name
		GPMsg.sigma_dim = self.sharedSigma.shape[0]
		GPMsg.mu_name = self.muMem.name
		GPMsg.mu_dim = self.sharedMu.shape[0]

		GPMsg.boundaries = self.boundaries

		self.GPPub.publish(GPMsg)

		self.GPPubTime = time.time() - GPPubStart

	def getMagCallback(self, request, response):
		mean, cov = self.magmap.predict(np.array(request.pos))

		response.mean = [float(mean[0]), float(mean[1]), float(mean[2])]
		response.cov = [float(cov[0]), float(cov[1]), float(cov[2])]

		return response

def main(args=None):
	rclpy.init(args=args)

	magmap = Magmap()
	
	try:
		rclpy.spin(magmap)
	except KeyboardInterrupt:
		print("Shutting down cleanly")

	magmap.log_file.close()
	print("Closed log file")

	magmap.SigmaMem.close()
	magmap.muMem.close()
	magmap.SigmaMem.unlink()
	magmap.muMem.unlink()
	print("Closed and unlinked shared memory")
	
	# Destroy the node
	magmap.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
