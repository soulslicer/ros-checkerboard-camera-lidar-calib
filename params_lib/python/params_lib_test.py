#!/usr/bin/env python
import glob
import os
import sys
import time
import random
import time
import numpy as np
import cv2
import sys
import os
import random
import math
import json
import rospy

import rospkg
devel_folder = rospkg.RosPack().get_path('params_lib').split("/")
devel_folder = '/'.join(devel_folder[:len(devel_folder)-2]) + "/devel/lib/"
print(devel_folder)
sys.path.append(devel_folder)
import params_lib_python

extrinsics_calibration = params_lib_python.ExtrinsicCalibration((3,3), 0.2)

K = np.array([810.80881, 0.0, 623.93996, 0.0, 809.94562, 539.02978, 0.0, 0.0, 1.0]).reshape((3,3))
T = np.eye(4)
Image = cv2.imread("/home/raaj/driver_ws/src/params_lib/python/after.png")

extrinsics_calibration.setCameraMatrix(K)

objPoints = extrinsics_calibration.calibrateExtrinsics(Image, T)

T = T.astype(np.float32)
objPoints = objPoints.astype(np.float32)

rosCloud = params_lib_python.convertXYZtoPointCloud2(objPoints)

xyzCloud = params_lib_python.convertPointCloud2toXYZ(rosCloud)

print(T)
print("---")
print(objPoints)
print("---")
print(xyzCloud)

cv2.imshow("win", Image)
cv2.waitKey(0)

# ROS Test
from sensor_msgs.msg import PointCloud2
import rospy
import time
rospy.init_node('params_lib_test', anonymous=True)
pub = rospy.Publisher('/cloud', PointCloud2, queue_size=1)
while 1:
    time.sleep(0.1)
    rosCloud.header.frame_id = "map"
    pub.publish(rosCloud)


