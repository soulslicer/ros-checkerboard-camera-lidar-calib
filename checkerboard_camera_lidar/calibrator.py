#!/usr/bin/python

import cv2
import message_filters
import numpy
import os
import rospy
import sensor_msgs.msg
import sensor_msgs.srv
import threading
import time
from camera_calibration_custom.calibrator import MonoCalibrator, StereoCalibrator, ChessboardInfo, Patterns, LidarCalibrator
from collections import deque
import image_geometry
from message_filters import ApproximateTimeSynchronizer
from std_msgs.msg import String
from std_srvs.srv import Empty
import functools
import time
from utils import *
path = rospkg.RosPack().get_path('checkerboard_camera_lidar') + "/"

import json
with open(path + 'datum.json') as json_file:
    datum = json.load(json_file)

class ConsumerThread(threading.Thread):
    def __init__(self, queue, function):
        threading.Thread.__init__(self)
        self.queue = queue
        self.function = function

    def run(self):
        while True:
            # wait for an image (could happen at the very beginning when the queue is still empty)
            while len(self.queue) == 0:
                time.sleep(0.1)
            self.function(self.queue[0])

from sensor_bridge.srv import *
class CalibrationNode:
    def __init__(self, datum):
        self.datum = datum
        self.c = None

        if self.datum["stereo"]:
            synchronizer = functools.partial(ApproximateTimeSynchronizer, slop=0.1)
            queue_size = 1
            left_camsub = message_filters.Subscriber(self.datum["left_cam"], sensor_msgs.msg.Image)
            left_caminfosub = message_filters.Subscriber(self.datum["left_cam_info"], sensor_msgs.msg.CameraInfo)
            right_camsub = message_filters.Subscriber(self.datum["right_cam"], sensor_msgs.msg.Image)
            right_caminfosub = message_filters.Subscriber(self.datum["right_cam_info"], sensor_msgs.msg.CameraInfo)
            lidarsub = message_filters.Subscriber(self.datum["lidar"], sensor_msgs.msg.PointCloud2)
            ts = synchronizer([left_camsub, left_caminfosub, right_camsub, right_caminfosub, lidarsub], queue_size)
            ts.registerCallback(self.queue_stereo)

            self.q_stereo = deque([], 1)
            lth = ConsumerThread(self.q_stereo, self.handle_data)
            lth.setDaemon(True)
            lth.start()
        else:
            synchronizer = functools.partial(ApproximateTimeSynchronizer, slop=0.1)
            queue_size = 1
            left_camsub = message_filters.Subscriber(self.datum["left_cam"], sensor_msgs.msg.Image)
            left_caminfosub = message_filters.Subscriber(self.datum["left_cam_info"], sensor_msgs.msg.CameraInfo)
            lidarsub = message_filters.Subscriber(self.datum["lidar"], sensor_msgs.msg.PointCloud2)
            ts = synchronizer([left_camsub, left_caminfosub, lidarsub], queue_size)
            ts.registerCallback(self.queue_mono)

            self.q_mono = deque([], 1)
            lth = ConsumerThread(self.q_mono, self.handle_data)
            lth.setDaemon(True)
            lth.start()


    def queue_stereo(self, left_cammsg, left_caminfomsg, right_cammsg, right_caminfomsg, lidarmsg):
        #print("Queue Stereo")
        self.q_stereo.append((left_cammsg, left_caminfomsg, right_cammsg, right_caminfomsg, lidarmsg))

    def queue_mono(self, left_cammsg, left_caminfomsg, lidarmsg):
        #print("Queue Mono")
        self.q_mono.append((left_cammsg, left_caminfomsg, lidarmsg))

    def handle_data(self, msg):
        #print("Handle Data")
        time.sleep(0.1)
        if self.c == None:
            self.c = LidarCalibrator(self.datum)

        self.c.handle_msg(msg)
        # self.displaywidth = drawable.scrib.shape[1]
        # self.redraw_monocular(drawable)


def main():
    rospy.init_node('node')
    calibNode = CalibrationNode(datum)
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()