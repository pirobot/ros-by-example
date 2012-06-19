#!/usr/bin/env python

""" face_detector.py - Version 1.0 2012-02-11

    Based on the OpenCV facedetect.py demo code
    
    Extends the ros2opencv2.py script which takes care of user input and image display
"""

import roslib; roslib.load_manifest('rbx_vision')
import rospy
import sys
import cv2
import cv2.cv as cv
from ros2opencv2 import ROS2OpenCV2
from sensor_msgs.msg import Image, RegionOfInterest

class FaceDetector(ROS2OpenCV2):
    def __init__(self, node_name):
        ROS2OpenCV2.__init__(self, node_name)
        
        self.node_name = node_name
        
        # Get the paths to the cascade XML files for the Haar detectors.
        # These are set in the launch file.
        cascade_1 = rospy.get_param("~cascade_frontal_alt", "")
        cascade_2 = rospy.get_param("~cascade_frontal_alt2", "")
        cascade_3 = rospy.get_param("~cascade_profile", "")
        
        # Initialize the Haar detectors using the cascade files
        self.cascade_1 = cv2.CascadeClassifier(cascade_1)
        self.cascade_2 = cv2.CascadeClassifier(cascade_2)
        self.cascade_3 = cv2.CascadeClassifier(cascade_3)

        # Set cascade classification parameters that tend to work well for faces
        self.haar_params = dict(minSize = (20, 20),
                                maxSize = (150, 150),
                                scaleFactor = 2.0,
                                minNeighbors = 1,
                                flags = cv.CV_HAAR_DO_CANNY_PRUNING)
        
        # Intialize the detection box
        self.detect_box = None
        
        # Initialize a couple of intermediate image variables
        self.grey = None
        self.small_image = None  
        
        # Track the number of hits and misses
        self.hits = 0
        self.misses = 0
        self.hit_rate = 0
        
        # Wait until the image topics are ready before starting
        rospy.wait_for_message("input_rgb_image", Image)
            
        rospy.loginfo("Ready.")

    def process_image(self, cv_image):
        # Create a greyscale version of the image
        grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
        # Attempt to detect a face
        self.detect_box = self.detect_face(grey)
        
        # Did we find one?
        if self.detect_box is not None:
            self.hits += 1
        else:
            self.misses += 1
        
        # Keep tabs on the hit rate so far
        self.hit_rate = float(self.hits) / (self.hits + self.misses)
                
        return cv_image

    def detect_face(self, input_image):
        # Equalize the histogram to reduce lighting effects
        search_image = cv2.equalizeHist(input_image)
        
        # Begin the search using three different XML template
        # First check one of the frontal templates
        if self.cascade_2:
            faces = self.cascade_2.detectMultiScale(search_image, **self.haar_params)
                                         
        # If that fails, check the profile template
        if not len(faces):
            if self.cascade_3:
                faces = self.cascade_3.detectMultiScale(search_image,**self.haar_params)

        # If that fails, check a different frontal profile
        if not len(faces):
            if self.cascade_1:
                faces = self.cascade_1.detectMultiScale(search_image, **self.haar_params)
        
        # If we did not detect any faces in this frame, display the message
        # "LOST FACE" on the marker image (defined in ros2opencv2.py)
        if not len(faces):
            self.last_face_box = None
            if self.show_text:
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(self.marker_image, "LOST FACE!", 
                            (20, int(self.frame_size[1] * 0.9)), 
                            font_face, font_scale, cv.RGB(255, 50, 50))
            return None
        
        # If the show_text is set, display the hit rate so far
        if self.show_text:
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            cv2.putText(self.marker_image, "Hit Rate: " + 
                        str(trunc(self.hit_rate, 2)), 
                        (20, int(self.frame_size[1] * 0.9)), 
                        font_face, font_scale, cv.RGB(255, 255, 0))
        
        # If we do have a face, rescale it and publish
        for (x, y, w, h) in faces:
            # Set the face box to be cvRect which is just a tuple in Python
            face_box = (x, y, w, h)
            
            # If we have a face, publish the bounding box as the ROI
            if face_box is not None:
                self.ROI = RegionOfInterest()
                self.ROI.x_offset = min(self.frame_size[0], max(0, x))
                self.ROI.y_offset = min(self.frame_size[1], max(0, y))
                self.ROI.width = min(self.frame_size[0], w)
                self.ROI.height = min(self.frame_size[1], h)
                
            self.pubROI.publish(self.ROI)

            # Break out of the loop after the first face 
            return face_box
        
def trunc(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    slen = len('%.*f' % (n, f))
    return float(str(f)[:slen])
    
if __name__ == '__main__':
    try:
      FaceDetector("face_detector")
      rospy.spin()
    except KeyboardInterrupt:
      print "Shutting down face detector node."
      cv2.destroyAllWindows()
