#!/usr/bin/env python


""" 

    hu_moments.py - Version 1.0 2012-03-05

"""

import roslib
roslib.load_manifest('pi_video_tracker')
import rospy
from ros2opencv2 import ROS2OpenCV2
from sensor_msgs.msg import Image, RegionOfInterest
from geometry_msgs.msg import Point
import sys
import cv2.cv as cv
import cv2
import numpy as np
from math import pow

class TemplateTracker(ROS2OpenCV2):
    def __init__(self, node_name):
        ROS2OpenCV2.__init__(self, node_name)
        
        self.match_threshold = rospy.get_param("~match_threshold", 0.7)
        self.find_multiple_targets = rospy.get_param("~find_multiple_targets", False)
        self.n_pyr = rospy.get_param("~n_pyr", 2)
        self.min_template_size = rospy.get_param("~min_template_size", 25)

        self.scale_factor = rospy.get_param("~scale_factor", 1.2)
        self.scale_and_rotate = rospy.get_param("~scale_and_rotate", False)
        
        self.use_depth_for_detection = rospy.get_param("~use_depth_for_detection", False)
        self.fov_width = rospy.get_param("~fov_width", 1.094)
        self.fov_height = rospy.get_param("~fov_height", 1.094)
        self.max_object_size = rospy.get_param("~max_object_size", 0.28)

        # Intialize the detection box
        self.detect_box = None
        
        # What kind of detector do we want to load
        self.detector_type = "template"
        self.detector_loaded = False
        
        rospy.loginfo("Waiting for video topics to become available...")

        # Wait until the image topics are ready before starting
        rospy.wait_for_message("input_rgb_image", Image)
        
        if self.use_depth_for_detection:
            rospy.wait_for_message("input_depth_image", Image)
            
        rospy.loginfo("Ready.")
        
    def process_image(self, cv_image):
        # Create a numpy array version of the image as required by many of the cv2 functions
        cv_array = np.array(cv_image, dtype=np.uint8)

        # Create a greyscale version of the image
        self.grey = cv2.cvtColor(cv_array, cv2.COLOR_BGR2GRAY)  
                
        # STEP 1. Load a detector if one is specified
        if self.detector_type and not self.detector_loaded:
            self.detector_loaded = self.load_detector(self.detector_type)
            
        # STEP 2: Detect the object
        #self.detect_box = self.detect_roi(self.detector_type, cv_image)
        
        if self.detect_box is not None:
            (template, roi) = self.get_template(self.detect_box, self.grey)
            self.track_box = self.detect_box
            self.detect_box = None

            moments = cv2.moments(template)
            hu_moments = cv2.HuMoments(moments)
            print hu_moments
                
        return cv_image
    
    def load_detector(self, detector):
        if detector == "template":
            #try:
            """ Read in the template image """              
            template_file = rospy.get_param("~template_file", "")
            
            self.template = cv2.imread(template_file, cv.CV_LOAD_IMAGE_COLOR)
            template_array = np.array(self.template, dtype = np.float32)
            template_grey = cv2.cvtColor(template_array, cv2.COLOR_RGB2GRAY)

            cv2.imshow("Template", self.template)
            
            template_moments = cv2.moments(template_grey)
            self.template_hu_moments = cv2.HuMoments(template_moments)
            
            print self.template_hu_moments
            
            return True
            #except:
                #rospy.loginfo("Exception loading face detector!")
                #return False
        else:
            return False
        
    def detect_roi(self, detector, cv_image):
        if detector == "template":
            detect_box = self.match_template(cv_image)
        
        return detect_box
    
    def get_template(self, roi, cv_array): 
        if len(roi) == 3:
            (center, size, angle) = roi
            pt1 = (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2))
            pt2 = (int(center[0] + size[0] / 2), int(center[1] + size[1] / 2))
            w = pt2[0] - pt1[0]
            h = pt2[1] - pt1[1]
            roi = (pt1[0], pt1[1], w, h)
            if w < 50 or h < 50:
                return (None, None)
        else:
            (x, y, w, h) = roi
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            roi = ((x + w/2, y + h/2), (w, h), 0)
        try:
            template = cv_array[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    
            cv.NamedWindow("New Template", cv.CV_NORMAL)
            cv.ResizeWindow("New Template", 320, 240)
            cv.MoveWindow("New Template", 700, 50)
            #cv.ShowImage("New Template", template)
            cv2.imshow("New Template", template)
            return (template, roi)
        except:
            rospy.loginfo("Exception getting template!")
            return (None, None)
    
    def match_template(self, cv_image):
        frame = np.array(cv_image, dtype=np.uint8)
        
        H,W = frame.shape[0], frame.shape[1]
        h,w = self.template.shape[0], self.template.shape[1]

        # Make sure that the template image is smaller than the source
        if W < w or H < h:
            rospy.loginfo( "Template image must be smaller than video frame." )
            return False
        
        if frame.dtype != self.template.dtype: 
            rospy.loginfo("Template and video frame must have same depth and number of channels.")
            return False
        
        # Create a copy of the frame to modify
        frame_copy = frame.copy()
        
        for i in range(self.n_pyr):
            frame_copy = cv2.pyrDown(frame_copy)
            
        template_height, template_width  = self.template.shape[:2]
        
        # Cycle through all scales starting with the last successful scale

        scales = self.scales[self.last_scale:] + self.scales[:self.last_scale - 1]

        # Track which scale and rotation gives the best match
        maxScore = -1
        best_s = 1
        best_r = 0
        best_x = 0
        best_y = 0
        
        for s in self.scales:
            for r in self.rotations:
                # Scale the template by s
                template_copy = cv2.resize(self.template, (int(template_width * s), int(template_height * s)))

                # Rotate the template through r degrees
                rotation_matrix = cv2.getRotationMatrix2D((template_copy.shape[1]/2, template_copy.shape[0]/2), r, 1.0)
                template_copy = cv2.warpAffine(template_copy, rotation_matrix, (template_copy.shape[1], template_copy.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    
                # Use pyrDown() n_pyr times on the scaled and rotated template
                for i in range(self.n_pyr):
                    template_copy = cv2.pyrDown(template_copy)
                
                # Create the results array to be used with matchTempate()
                h,w = template_copy.shape[:2]
                H,W = frame_copy.shape[:2]
                
                result_width = W - w + 1
                result_height = H - h + 1
                
                try:
                    result_mat = cv.CreateMat(result_height, result_width, cv.CV_32FC1)
                    result = np.array(result_mat, dtype = np.float32)
                except:
                    continue
                
                # Run matchTemplate() on the reduced images
                cv2.matchTemplate(frame_copy, template_copy, cv.CV_TM_CCOEFF_NORMED, result)
                
                # Find the maximum value on the result map
                (minValue, maxValue, minLoc, maxLoc) = cv2.minMaxLoc(result)
                
                if maxValue > maxScore:
                    maxScore = maxValue
                    best_x, best_y = maxLoc
                    best_s = s
                    best_r = r
                    best_template = template_copy.copy()
                    self.last_scale = self.scales.index(s)
                    best_result = result.copy()
                
        # Transform back to original image sizes
        best_x *= int(pow(2.0, self.n_pyr))
        best_y *= int(pow(2.0, self.n_pyr))
        h,w = self.template.shape[:2]
        h = int(h * best_s)
        w = int(w * best_s)
        best_result = cv2.resize(best_result, (int(pow(2.0, self.n_pyr)) * best_result.shape[1], int(pow(2.0, self.n_pyr)) * best_result.shape[0]))
        cv2.imshow("Result", best_result)
        best_template = cv2.resize(best_template, (int(pow(2.0, self.n_pyr)) * best_template.shape[1], int(pow(2.0, self.n_pyr)) * best_template.shape[0]))
        cv2.imshow("Best Template", best_template)
        
        #match_box = ((best_x + w/2, best_y + h/2), (w, h), -best_r)
        return (best_x, best_y, w, h)
    
    def match_template1(self, cv_image):
        frame = np.array(cv_image, dtype=np.uint8)
        
        H,W = frame.shape[0], frame.shape[1]
        h,w = self.template.shape[0], self.template.shape[1]

        # Make sure that the template image is smaller than the source
        if W < w or H < h:
            rospy.loginfo( "Template image must be smaller than video frame." )
            return False
        
        if frame.dtype != self.template.dtype: 
            rospy.loginfo("Template and video frame must have same depth and number of channels.")
            return False
        
        # Create a copy of the frame to modify
        frame_copy = frame.copy()
        
        for i in range(self.n_pyr):
            frame_copy = cv2.pyrDown(frame_copy)
            
        template_height, template_width  = self.template.shape[:2]
        
        # Cycle through all scales starting with the last successful scale

        scales = self.scales[self.last_scale:] + self.scales[:self.last_scale - 1]

        # Track which scale and rotation gives the best match
        maxScore = -1
        best_s = 1
        best_r = 0
        best_x = 0
        best_y = 0
        
        for s in self.scales:
            for r in self.rotations:
                # Scale the template by s
                template_copy = cv2.resize(self.template, (int(template_width * s), int(template_height * s)))

                # Rotate the template through r degrees
                rotation_matrix = cv2.getRotationMatrix2D((template_copy.shape[1]/2, template_copy.shape[0]/2), r, 1.0)
                template_copy = cv2.warpAffine(template_copy, rotation_matrix, (template_copy.shape[1], template_copy.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    
                # Use pyrDown() n_pyr times on the scaled and rotated template
                for i in range(self.n_pyr):
                    template_copy = cv2.pyrDown(template_copy)
                
                # Create the results array to be used with matchTempate()
                h,w = template_copy.shape[:2]
                H,W = frame_copy.shape[:2]
                
                result_width = W - w + 1
                result_height = H - h + 1
                
                try:
                    result_mat = cv.CreateMat(result_height, result_width, cv.CV_32FC1)
                    result = np.array(result_mat, dtype = np.float32)
                except:
                    continue
                
                # Run matchTemplate() on the reduced images
                cv2.matchTemplate(frame_copy, template_copy, cv.CV_TM_CCOEFF_NORMED, result)
                
                # Find the maximum value on the result map
                (minValue, maxValue, minLoc, maxLoc) = cv2.minMaxLoc(result)
                
                if maxValue > maxScore:
                    maxScore = maxValue
                    best_x, best_y = maxLoc
                    best_s = s
                    best_r = r
                    best_template = template_copy.copy()
                    self.last_scale = self.scales.index(s)
                    best_result = result.copy()
                
        # Transform back to original image sizes
        best_x *= int(pow(2.0, self.n_pyr))
        best_y *= int(pow(2.0, self.n_pyr))
        h,w = self.template.shape[:2]
        h = int(h * best_s)
        w = int(w * best_s)
        best_result = cv2.resize(best_result, (int(pow(2.0, self.n_pyr)) * best_result.shape[1], int(pow(2.0, self.n_pyr)) * best_result.shape[0]))
        cv2.imshow("Result", best_result)
        best_template = cv2.resize(best_template, (int(pow(2.0, self.n_pyr)) * best_template.shape[1], int(pow(2.0, self.n_pyr)) * best_template.shape[0]))
        cv2.imshow("Best Template", best_template)
        
        #match_box = ((best_x + w/2, best_y + h/2), (w, h), -best_r)
        return (best_x, best_y, w, h)

def main(args):
    PMT = TemplateTracker("template_tracker")
    try:
      rospy.spin()
    except KeyboardInterrupt:
      print "Shutting down fast match template node."
      cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    