## Checkerboard Camera Lidar Calib

This ROS module lets you use a normal checkerboard to calibrate a camera or stereo camera pair to a lidar. The defautlt settings used are for an OS2-128 Lidar, so you may need to change some parameters. This relies on ROS, PCL and Pytorch.

### Running

"""
1. You can download a sample bag here
	https://drive.google.com/file/d/1f5zYcTOGtbYByyyUUwLuK0yGBAvSxEKL/view?usp=sharing
2. rosbag play --clock calib_sample.bag
3. Change the parameters in datum.json
4. rosrun checkerboard_camera_lidar calibrator.py
5. rosrun checkerboard_camera_lidar lidar_calib.py
"""
