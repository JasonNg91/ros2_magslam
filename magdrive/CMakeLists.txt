cmake_minimum_required(VERSION 3.5)
project(magdrive)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)

#add executables
add_executable(driver src/driver.cpp)
ament_target_dependencies(driver rclcpp geometry_msgs tf2)

install(TARGETS
  driver
  DESTINATION lib/${PROJECT_NAME})


ament_package()
