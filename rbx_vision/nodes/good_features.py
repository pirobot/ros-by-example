#!/usr/bin/env python

""" good_features.py - Version 1.0 2012-02-11

    Locate the Good Features To Track in a video stream.
"""

import roslib; roslib.load_manifest('rbx_vision')
import rospy
from ros2opencv2 import ROS2OpenCV2
import sys
import cv2
import cv2.cv as cv
import numpy as np

class GoodFeatures(ROS2OpenCV2):
    def __init__(self, node_name):
        ROS2OpenCV2.__init__(self, node_name)
        
        self.node_name = node_name
        
        self.show_text = rospy.get_param("~show_text", True)
        self.feature_size = rospy.get_param("~feature_size", 1)
        
        # Good features parameters
        self.maxCorners = rospy.get_param("~maxCorners", 200)
        self.qualityLevel = rospy.get_param("~qualityLevel", 0.01)
        self.minDistance = rospy.get_param("~minDistance", 7)
        self.blockSize = rospy.get_param("~blockSize", 10)
        self.useHarrisDetector = rospy.get_param("~useHarrisDetector", False)
        self.k = rospy.get_param("~k", 0.04)
        self.flip_image = rospy.get_param("~flip_image", False)
        
        self.gf_params = dict( maxCorners = self.maxCorners, 
                       qualityLevel = self.qualityLevel,
                       minDistance = self.minDistance,
                       blockSize = self.blockSize,
                       useHarrisDetector = self.useHarrisDetector,
                       k = self.k )

        self.detect_interval = 1
        self.features = []
        self.frame_idx = 0

        self.detect_box = None
        self.mask = None
        

    def process_image(self, cv_image):
        if not self.detect_box:
            return cv_image

        # Create a greyscale version of the image
        self.grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        """ Redetect features at the given interval """        
        if self.frame_idx % self.detect_interval == 0:
            self.keypoints = []
            self.get_keypoints(self.detect_box)
        
        # Process any special keyboard commands for this module
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 'c':
                self.keypoints = []
                self.detect_box = None
                
        self.frame_idx += 1
                
        return cv_image

    def get_keypoints(self, detect_box):
        """ Zero the mask with all black pixels """
        self.mask = np.zeros_like(self.grey)
 
        """ Get the coordinates and dimensions of the track box """
        try:
            x,y,w,h = detect_box
        except: 
            return None

        """ For manually selected regions, just use a rectangle """
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        
        """ Set the rectangule within the mask to white """
        self.mask[y:y+h, x:x+w] = 255

        p = cv2.goodFeaturesToTrack(self.grey, mask = self.mask, **self.gf_params)
        
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.keypoints.append([(x, y)])
                cv2.circle(self.marker_image, (x, y), self.feature_size, (0, 255, 0, 0), cv.CV_FILLED, 8, 0)    

def main(args):
      GoodFeatures("good_features")
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print "Shutting down the Good Features node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
