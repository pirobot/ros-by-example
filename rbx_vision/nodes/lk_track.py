#!/usr/bin/env python

""" lk_track.py - Version 1.0 2012-02-11

    Based on the OpenCV lk_track.py demo code 
"""

import roslib; roslib.load_manifest('rbx_vision')
import rospy
from ros2opencv2 import ROS2OpenCV2
import sys
import cv2
import cv2.cv as cv
from sensor_msgs.msg import Image, RegionOfInterest
import numpy as np
from time import clock

class LKTrack(ROS2OpenCV2):
    def __init__(self, node_name):
        ROS2OpenCV2.__init__(self, node_name)
        
        self.node_name = node_name
        
        self.show_text = rospy.get_param("~show_text", True)
        self.feature_size = rospy.get_param("~feature_size", 2)
        
        # Good Feature paramters
        self.maxCorners = rospy.get_param("~maxCorners", 200)
        self.qualityLevel = rospy.get_param("~qualityLevel", 0.01)
        self.minDistance = rospy.get_param("~minDistance", 5)
        self.blockSize = rospy.get_param("~blockSize", 3)
        self.useHarrisDetector = rospy.get_param("~useHarrisDetector", False)
        self.k = rospy.get_param("~k", 0.04)
        self.flip_image = rospy.get_param("~flip_image", False)
        
        # LK parameters
        self.winSize = rospy.get_param("~winSize", (10, 10))
        self.maxLevel = rospy.get_param("~maxLevel", 2)
        self.criteria = rospy.get_param("~criteria", (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.derivLambda = rospy.get_param("~derivLambda", 0.1)

        self.gf_params = dict( maxCorners = self.maxCorners, 
                       qualityLevel = self.qualityLevel,
                       minDistance = self.minDistance,
                       blockSize = self.blockSize,
                       useHarrisDetector = self.useHarrisDetector,
                       k = self.k )
        
        self.lk_params = dict( winSize  = self.winSize, 
                  maxLevel = self.maxLevel, 
                  criteria = self.criteria,
                  derivLambda = self.derivLambda )    
        
        self.detect_interval = 1
        self.keypoints = []

        self.detect_box = None
        self.track_box = None
        self.mask = None
        self.prev_grey = None
        
        rospy.loginfo("Waiting for video topics to become available...")

        # Wait until the image topics are ready before starting
        rospy.wait_for_message("input_rgb_image", Image)
            
        rospy.loginfo("Ready.")

    def process_image(self, cv_image):
        """ Step 1: If we don't yet have a detection box (drawn by the user with the mouse), keep waiting. """
        if self.detect_box is None:
            return cv_image
                
        # Create a numpy array version of the image as required by many of the cv2 functions
        cv_array = np.array(cv_image, dtype=np.uint8)

        # Create a greyscale version of the image
        self.grey = cv2.cvtColor(cv_array, cv2.COLOR_BGR2GRAY)
        
        # Equalize the grey histogram to minimize lighting effects
        self.grey = cv2.equalizeHist(self.grey)
        
        """ Step 2: If we haven't yet started tracking, initialize the track box to be the detect box and 
                    extract the keypoints within it. """
        if self.track_box is None or not self.is_rect_nonzero(self.track_box):
            self.track_box = self.detect_box
            self.keypoints = []
            self.get_keypoints(self.track_box)
            
        if self.prev_grey is None:
            self.prev_grey = self.grey

        """ Step 3:  Now that have keypoints, track them to the next frame using optical flow. """
        self.track_box = self.track_keypoints()
        
        # Process any special keyboard commands for this module
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 'c':
                self.keypoints = []
                self.track_box = None
                self.detect_box = None
                self.classifier_initialized = True
                
        self.prev_grey = self.grey
                
        return cv_image

    def get_keypoints(self, track_box):
        """ Zero the mask with all black pixels """
        mask = np.zeros_like(self.grey)
     
        """ Get the coordinates and dimensions of the track box """
        try:
            x,y,w,h = track_box
        except:
            return None
        
        mask_box = ((x + w/2, y + h/2), (w, h), 0)
        
        cv2.ellipse(mask, mask_box, cv.RGB(255, 255, 255), -1)
        
        #""" Set the rectangule within the mask to white """
        #mask[y:y+h, x:x+w] = 255
                
        for x, y in [np.int32(p[-1]) for p in self.keypoints]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(self.grey, mask = mask, **self.gf_params)
        
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.keypoints.append((x, y))
                cv2.circle(self.marker_image, (x, y), self.feature_size, (0, 255, 0, 0), cv.CV_FILLED, 8, 0)                
                    
    def track_keypoints(self):
        if len(self.keypoints) > 0:
            img0, img1 = self.prev_grey, self.grey
            p0 = np.float32([p for p in self.keypoints]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.   lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_keypoints = []
            for p, (x, y), good_flag in zip(self.keypoints, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                new_keypoints.append((x, y))
                cv2.circle(self.marker_image, (x, y), self.feature_size, (0, 255, 0, 0), cv.CV_FILLED, 8, 0)
            self.keypoints = new_keypoints
            
        """ Draw the best fit ellipse around the feature points """
        if len(self.keypoints) > 6:
            self.keypoints_matrix = cv.CreateMat(1, len(self.keypoints), cv.CV_32SC2)
            i = 0
            for p in self.keypoints:
                cv.Set2D(self.keypoints_matrix, 0, i, (int(p[0]), int(p[1])))
                i = i + 1           
            keypoints_box = cv.FitEllipse2(self.keypoints_matrix)
            #keypoints_box = cv.BoundingRect(self.keypoints)
        else:
            keypoints_box = None
        
        return keypoints_box
        
    
def main(args):
      LK = LKTrack("lk_track")
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print "Shutting down LK Tracking node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    