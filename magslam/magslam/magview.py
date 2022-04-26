from tkinter import W
import rclpy
from rclpy.node import Node

from magslam_msgs.msg import MagGPMem

from .gp.magfield import GPMagMap

from tf2_ros import TransformException
from tf2_ros import TransformListener
from tf2_ros.buffer import Buffer

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from .gp.DGP.plotfunctions import transparencyHeatMap
import time

from multiprocessing import shared_memory
import threading

import os

class Magview(Node):

	def __init__(self):
		super().__init__('magview')

		# Parameter setup
		self.declare_parameter('sensor_frame', 'imu_link')
		self.sensorFrame = self.get_parameter('sensor_frame').get_parameter_value().string_value
		self.get_logger().info('sensor_frame: ' + self.sensorFrame)
		
		self.declare_parameter('world_frame', 'odom')
		self.worldFrame = self.get_parameter('world_frame').get_parameter_value().string_value
		self.get_logger().info('world_frame: ' + self.worldFrame)

		self.gp_sub = self.create_subscription(
			MagGPMem,
			'/magmap/gp',
			self.updateGPMem,
			1)

		# Transform listener for path plot
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.path = np.zeros(shape=(1,2))
		
		# Variable to know if a GP has already been received
		self.GPReady = False

		#Figure and plot
		self.fig = plt.figure('Magmap',figsize=(12,12))
		self.ax = self.fig.add_subplot(111,)

		# Timer for updating map and timer for updating path
		self.plotTimer = self.create_timer(1.0, self.plot)
		self.pathTimer = self.create_timer(0.1, self.updatePath)

		# Frame number for png saving
		self.framenr = 0

		# Create folder for png saving
		if (not os.path.exists('mapframes/nopath')):
			os.makedirs('mapframes/nopath/')

		# Map properties
		self.minVals = [-2.0, -2.0]
		self.maxVals = [2.0, 2.0]

		z = 0.0

		w = 60
		h = 60

		linX = np.linspace(self.minVals[0], self.maxVals[0], w)
		linY = np.linspace(self.minVals[1], self.maxVals[1], h)

		self.pos_xy = np.hstack([linX[:,None],linY[:,None]])

		xx, yy, zz = np.meshgrid(linX, linY, z)
		self.positions = jnp.array(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).transpose())

		self.predictd_seq = np.zeros((2,w,h))
		self.mapThread = threading.Thread(target=self.updateMap)

		self.get_logger().info('Magview node initiated.')

		self.startTime = time.time()

	# Plot and save magmap with and without path
	def plot(self):

		self.ax.clear()
		transparencyHeatMap(self.minVals, self.maxVals, self.pos_xy, self.predictd_seq, self.fig, self.ax)

		imTime = time.time() - self.startTime
		
		plt.savefig('mapframes/nopath/frame' + str(self.framenr) + "_" + str(imTime) + '.png', transparent=True)

		self.ax.plot(self.path[1:,0],self.path[1:,1],color=(0.5, 0.5, 0.5, 0.5))

		plt.pause(0.01)
		plt.savefig('mapframes/frame' + str(self.framenr) + "_" + str(imTime) + '.png', transparent=True)

		self.framenr += 1

	# Retrieve GP from shared memory
	def updateGPMem(self, gp_msg):
		if (self.GPReady):
			if (self.mapThread.is_alive() == False):
				self.SigmaMem = shared_memory.SharedMemory(name=gp_msg.sigma_name)
				self.muMem = shared_memory.SharedMemory(name=gp_msg.mu_name)
				
				self.magmap.Sigma = np.ndarray((gp_msg.sigma_dim,gp_msg.sigma_dim), dtype=np.float64, buffer=self.SigmaMem.buf)
				self.magmap.mu = np.ndarray((gp_msg.mu_dim,), dtype=np.float64, buffer=self.muMem.buf)

				self.mapThread = threading.Thread(target=self.updateMap)
				self.mapThread.start()
		else:
			self.magmap = GPMagMap(gp_msg.boundaries, gp_msg.m_basis, gp_msg.lin_var, gp_msg.stat_var, gp_msg.stat_ls, gp_msg.lik_var)
			self.GPReady = True
			self.get_logger().info('GP Initiated.')
			self.updateGPMem(gp_msg)

	# Calculate new magnetic field estimates
	def updateMap(self):
		print("Map update thread started")

		start = time.time()
		self.predictd_seq = self.magmap.model_seq.predict_seq_vmap(self.positions, self.magmap.mu, self.magmap.Sigma)
		end = time.time()

		print("Map update done, elapsed time (s):{}".format(end - start))

	# Add last known position to traveled path
	def updatePath(self):
		try:
			now = rclpy.time.Time()
			trans = self.tf_buffer.lookup_transform(self.worldFrame, self.sensorFrame, now)
			print("Position Added")
		except TransformException as ex:
			self.get_logger().warn(f'Could not transform {self.sensorFrame} to {self.worldFrame}: {ex}')
			return

		self.path = np.vstack((self.path, [trans.transform.translation.x, trans.transform.translation.y]))

def main(args=None):
	rclpy.init(args=args)

	magview = Magview()

	# Run node till keyboard interrupt
	try:
		rclpy.spin(magview)
	except KeyboardInterrupt:
		print("Shutting down cleanly")

	# Close shared memory
	magview.SigmaMem.close()
	magview.muMem.close()
	print("Closed shared memory")

	# Destroy the node
	magview.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
