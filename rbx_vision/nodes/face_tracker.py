#!/usr/bin/env python

""" face_tracker.py - Version 1.0 2012-02-11

    Combines the OpenCV Haar face detector with Good Features to Track and Lucas-Kanade
    optical flow tracking.
     
"""

import roslib
roslib.load_manifest('rbx_vision')
import rospy
import cv2
import cv2.cv as cv
from ros2opencv2 import ROS2OpenCV2
from lk_tracker import LKTracker
from face_detector import FaceDetector
import numpy as np

class FaceTracker(FaceDetector, LKTracker):
    def __init__(self, node_name):
        FaceDetector.__init__(self, node_name)
        LKTracker.__init__(self, node_name)
            
        self.n_faces = rospy.get_param("~n_faces", 1)
        self.show_text = rospy.get_param("~show_text", True)
        self.feature_size = rospy.get_param("~feature_size", 1)
        
        self.use_depth_for_detection = rospy.get_param("~use_depth_for_detection", False)
        self.fov_width = rospy.get_param("~fov_width", 1.094)
        self.fov_height = rospy.get_param("~fov_height", 1.094)
        self.max_object_size = rospy.get_param("~max_face_size", 0.28)

        self.detect_interval = 1
        self.keypoints = list()

        self.detect_box = None
        self.track_box = None
        
        self.grey = None
        self.prev_grey = None
    
    def process_image(self, cv_image):
        """ STEP 0: Preprocess the image """
        # Create a greyscale version of the image
        self.grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Equalize the grey histogram to minimize lighting effects
        self.grey = cv2.equalizeHist(self.grey)
        
        """ STEP 1: Detect the face if we haven't already """
        if self.detect_box is None:
            self.detect_box = self.detect_face(self.grey)
        
        """ Step 2: Extract keypoints """
        if self.track_box is None or not self.is_rect_nonzero(self.track_box):
            self.track_box = self.detect_box
            self.keypoints = list()
            self.keypoints = self.get_keypoints(self.grey, self.track_box)
            
        if self.prev_grey is None:
            self.prev_grey = self.grey
            
        """ Step 3:  Track keypoints using optical flow """
        self.track_box = self.track_keypoints(self.grey, self.prev_grey)
        
        # Process any special keyboard commands for this module
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 'c':
                self.keypoints = []
                self.track_box = None
                self.detect_box = None
                
        self.prev_grey = self.grey
                
        return cv_image                
    
if __name__ == '__main__':
    try:
        node_name = "face_tracker"
        FaceTracker(node_name)
        try:
            rospy.init_node(node_name)
        except:
            pass
        rospy.loginfo("Starting node " + str(node_name))
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down face tracker node."
        cv.DestroyAllWindows()
