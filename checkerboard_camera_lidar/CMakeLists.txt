cmake_minimum_required(VERSION 2.8)
project(checkerboard_camera_lidar)

find_package(catkin REQUIRED)
catkin_package()

catkin_python_setup()

catkin_install_python(PROGRAMS calibrator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
