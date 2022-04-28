#include <memory>
#include <iostream>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

using std::placeholders::_1;

#define PI 3.14159265358979323846

/// Function to generate a grid path
std::vector<std::vector<float>> generateGrid(float xmin, float ymin, float xmax, float ymax, float gridsize, float sensorY){
	std::vector<std::vector<float>> waypoints;

	bool atMin = true;

	for (float y = ymin; y <= ymax; y += gridsize){
		if (atMin){
			waypoints.push_back({xmin - sensorY, y + sensorY});
			waypoints.push_back({xmax - sensorY, y + sensorY});
			atMin = false;
		} else {
			waypoints.push_back({xmax - sensorY, y - sensorY});
			waypoints.push_back({xmin - sensorY, y - sensorY});
			atMin = true;
		}
	}

	atMin = false;

	for (float x = xmax; x >= xmin; x -= gridsize){
		if (atMin){
			waypoints.push_back({x - sensorY, ymin - sensorY});
			waypoints.push_back({x - sensorY, ymax - sensorY});
			atMin = false;
		} else {
			waypoints.push_back({x + sensorY, ymax - sensorY});
			waypoints.push_back({x + sensorY, ymin - sensorY});
			atMin = true;
		}
	}

	return waypoints;
}

class Driver : public rclcpp::Node
{
public:
	Driver()
	: Node("driver") {
		RCLCPP_INFO(this->get_logger(), "Magdrive starting");
		
		// Subscribe to positions from OptiTrack
		pose_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
		"Robot_1/pose", 10, std::bind(&Driver::pose_callback, this, _1));
		RCLCPP_INFO(this->get_logger(), "Setup subscriber");
		
		// Publisher for robot control
		cmd_pub = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
		RCLCPP_INFO(this->get_logger(), "Setup publisher");

		// Parameters
		RCLCPP_INFO(this->get_logger(), "Retrieving parameters");

		std::string pathMode;
		float xMin;
		float yMin;
		float xMax;
		float yMax;
		float density;
		float sensorY;

		this->declare_parameter<std::string>("path_mode", "rectangle");
		this->declare_parameter<float>("x_min", -1.0);
		this->declare_parameter<float>("y_min", -1.0);
		this->declare_parameter<float>("x_max", 1.0);
		this->declare_parameter<float>("y_max", 1.0);
		this->declare_parameter<float>("sensor_y", 0.0);

		this->get_parameter("path_mode", pathMode);
		this->get_parameter("x_min", xMin);
		this->get_parameter("y_min", yMin);
		this->get_parameter("x_max", xMax);
		this->get_parameter("y_max", yMax);
		this->get_parameter("density", density);
		this->get_parameter("sensor_y", sensorY);
		
		RCLCPP_INFO(this->get_logger(), "path_mode: %s", pathMode.c_str());
		RCLCPP_INFO(this->get_logger(), "x_min: %.2f", xMin);
		RCLCPP_INFO(this->get_logger(), "y_min: %.2f", yMin);
		RCLCPP_INFO(this->get_logger(), "x_max: %.2f", xMax);
		RCLCPP_INFO(this->get_logger(), "y_max: %.2f", yMax);
		RCLCPP_INFO(this->get_logger(), "density: %.2f", density);
		RCLCPP_INFO(this->get_logger(), "sensor_y: %.2f", sensorY);

		// Calculate path
		if (pathMode == "rectangle") {
			//Square path
			waypoints = {
				{xMax - sensorY, yMax - sensorY}, 
				{xMin + sensorY, yMax - sensorY}, 
				{xMin + sensorY, yMin + sensorY}, 
				{xMax - sensorY, yMin + sensorY}
			};
		}
		else if (pathMode == "rectangle_invert") {
			//Square path, testing in opposite direction
			waypoints = {
				{xMax - sensorY, yMax - sensorY}, 
				{xMin + sensorY, yMax - sensorY}, 
				{xMin + sensorY, yMin + sensorY}, 
				{xMax - sensorY, yMin + sensorY},
				{xMax + sensorY, yMax + sensorY}, 
				{xMax + sensorY, yMin - sensorY},
				{xMin - sensorY, yMin - sensorY}, 
				{xMin - sensorY, yMax + sensorY}
			};
		}
		else if (pathMode == "grid"){
			//Grid path
			waypoints = generateGrid(xMin, yMin, xMax, yMax, density, sensorY);
		}
		else {
			RCLCPP_ERROR(this->get_logger(), "Invalid path_mode");

		}
	}

private:
	std::vector<std::vector<float>> waypoints;
	
	int curTarget = 0;			// Index of current target in waypoints vector

	// When a position is received send move commands to robot
	void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr pose_msg){
		// Get orientation and convert to RPY via rotation matrix
		tf2::Quaternion q(pose_msg->pose.orientation.x, pose_msg->pose.orientation.y, pose_msg->pose.orientation.z, pose_msg->pose.orientation.w);
		tf2::Matrix3x3 m(q);

		double roll, pitch, yaw;
		m.getRPY(roll, pitch, yaw);

		// Deviation from target position
		const float dx = waypoints[curTarget][0] - pose_msg->pose.position.x;
		const float dy = waypoints[curTarget][1] - pose_msg->pose.position.y;;
		
		// Target direction pointing to target position
		const double targetYaw=atan2(dy, dx);
		
		// Deviation from target yaw
		double yawDiff = targetYaw - yaw;
		if (yawDiff < -PI) yawDiff += 2*PI;
		if (yawDiff > PI) yawDiff -= 2*PI;

		geometry_msgs::msg::Twist cmd_msg;

		// Calculate turning speed
		cmd_msg.angular.z = yawDiff * 3.0;
		if (cmd_msg.angular.z > 1.0) cmd_msg.angular.z = 1.0;
		if (cmd_msg.angular.z < -1.0) cmd_msg.angular.z = -1.0;

		// Distance to target
		const float distance = sqrt(pow(dx, 2) + pow(dy, 2));
		
		// Drive forward if direction is about correct
		if (abs(yawDiff) < 0.1){
			cmd_msg.linear.x = distance * 2.0;

			if (cmd_msg.linear.x > 0.5)	cmd_msg.linear.x = 0.5;		// Limit linear speed
		}

		// If the target is reached set next target
		if (distance < 0.025){
			curTarget++;
			if (curTarget >= waypoints.size()) curTarget = 0;
		}
		
		cmd_pub->publish(cmd_msg);
	}

	rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub;
	rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub;
};

int main(int argc, char * argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<Driver>());
	rclcpp::shutdown();
	return 0;
}
