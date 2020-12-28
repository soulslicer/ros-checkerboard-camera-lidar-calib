## Checkerboard Camera Lidar Calib

This ROS module lets you use a normal checkerboard to calibrate a camera or stereo camera pair to a lidar. The defautlt settings used are for an OS2-128 Lidar, so you may need to change some parameters. This relies on ROS, PCL and Pytorch.

![Upsampling](https://github.com/soulslicer/ros-checkerboard-camera-lidar-calib/blob/main/img/1.png?raw=true)

### Setup

```
mkdir catkin_ws; cd catkin_ws
git clone https://github.com/soulslicer/ros-checkerboard-camera-lidar-calib.git src
catkin_make -DCMAKE_BUILD_TYPE=Release
```

### Running

```
1. You can download a sample bag here
	https://drive.google.com/file/d/1f5zYcTOGtbYByyyUUwLuK0yGBAvSxEKL/view?usp=sharing
2. rosbag play --clock calib_sample.bag
3. Change the parameters in datum.json
4. rosrun checkerboard_camera_lidar calibrator.py
	Press space on rosbag to pause it
	Press space on the opencv window to store that datapoint. Ensure that the board edge and lidar edge colors are correctly matched
	With enough data, press w to save it
5. rosrun checkerboard_camera_lidar lidar_calib.py
```

### Parameters

```
{
   # Board
   "dim":0.19, # Checkerboard square Dimensions
   "cols":4, # Checkerboard size
   "rows":5,
   "offsets":[ # TL, TR, BL, BR offsets from physical board edge to actual checkerboard. Approx value is fine exact not needed. Assists in edge search
      0.02,
      0.02,
      0.06,
      0.03,
      0.02,
      0.02,
      0.06,
      0.03
   ],
   "board_area":1.2 # Physical area of actual board
   "minimum_zdist":2.0, # If board is closer than 2m we ignore it
   
   # Lidar
   "lidar_tilt":7.0, # Pitch of the Lidar
   "lidar_height":1.9, # How high the lidar is from the ground (Used to remove ground plane)
   "lidar_to_left_cam":[ # Initial guess. This is important!
      0.5,
      -0.25,
      0,
      1.5692,
      0.0013931,
      1.5941
   ],
   
   # ROS
   "stereo":false, # Stereo Mode or not
   "left_cam":"/left_camera/image_color",
   "left_cam_info":"/left_camera/camera_info",
   "right_cam":"/right_camera/image_color",
   "right_cam_info":"/right_camera/camera_info",
   "lidar":"/lidar0/points",

   # Parameters Additional
   "plane_fit_ransac_error":0.05,
   "line_fit_ransac_error":0.02,
   "minimum_area_diff":0.1,
   
   "save_location":"/tmp/data/",
}
```
