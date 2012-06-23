#!/usr/bin/env python

""" lk_tracker.py - Version 1.0 2012-02-11

    Based on the OpenCV lk_track.py demo code 
"""

import roslib; roslib.load_manifest('rbx_vision')
import rospy
import cv2
import cv2.cv as cv
from good_features import GoodFeatures
from sensor_msgs.msg import Image, RegionOfInterest
import numpy as np

class LKTracker(GoodFeatures):
    def __init__(self, node_name):
        GoodFeatures.__init__(self, node_name)
        
        self.show_text = rospy.get_param("~show_text", True)
        self.feature_size = rospy.get_param("~feature_size", 1)
        
        # LK parameters
        self.winSize = rospy.get_param("~winSize", (10, 10))
        self.maxLevel = rospy.get_param("~maxLevel", 2)
        self.criteria = rospy.get_param("~criteria", (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.derivLambda = rospy.get_param("~derivLambda", 0.1)
        
        self.lk_params = dict( winSize  = self.winSize, 
                  maxLevel = self.maxLevel, 
                  criteria = self.criteria,
                  derivLambda = self.derivLambda )    
        
        self.detect_interval = 1
        self.keypoints = list()

        self.detect_box = None
        self.track_box = None
        self.mask = None
        self.grey = None
        self.prev_grey = None
            
    def process_image(self, cv_image):
        # STEP 1: If we don't yet have a detection box (drawn by the user with the mouse), keep waiting
        if self.detect_box is None:
            return cv_image

        # Create a greyscale version of the image
        self.grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Equalize the grey histogram to minimize lighting effects
        self.grey = cv2.equalizeHist(self.grey)
        
        # STEP 2: If we haven't yet started tracking, set the track box to the
        #         detect box and extract the keypoints within it
        if self.track_box is None or not self.is_rect_nonzero(self.track_box):
            self.track_box = self.detect_box
            self.keypoints = list()
            self.keypoints = self.get_keypoints(self.grey, self.track_box)
        
        else:
            if self.prev_grey is None:
                self.prev_grey = self.grey
    
            # STEP 3:  Now that have keypoints, track them to the next frame using optical flow
            self.track_box = self.track_keypoints(self.grey, self.prev_grey)

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
                    
    def track_keypoints(self, grey, prev_grey):
        if len(self.keypoints) > 0:
            img0, img1 = prev_grey, grey
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
        
    
if __name__ == '__main__':
    try:
        node_name = "lk_tracker"
        LKTracker(node_name)
        try:
            rospy.init_node(node_name)
        except:
            pass
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down LK Tracking node."
        cv.DestroyAllWindows()
    