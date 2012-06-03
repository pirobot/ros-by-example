#!/usr/bin/env python

""" face_detector.py - Version 1.0 2012-02-11

    Based on the OpenCV facedetect.py demo code 
"""

import roslib
roslib.load_manifest('rbx_vision')
import rospy
from ros2opencv2 import ROS2OpenCV2
import sys
import cv2.cv as cv
import cv2
from sensor_msgs.msg import Image, RegionOfInterest

class FaceDetector(ROS2OpenCV2):
    def __init__(self, node_name):
        ROS2OpenCV2.__init__(self, node_name)
        
        self.node_name = node_name

        self.use_depth_for_detection = rospy.get_param("~use_depth_for_detection", False)
        self.fov_width = rospy.get_param("~fov_width", 1.094)
        self.fov_height = rospy.get_param("~fov_height", 1.094)
        self.max_face_size = rospy.get_param("~max_face_size", 0.28)
        self.use_last_face_box = rospy.get_param("~use_last_face_box", False)
        
        # Intialize the detection box
        self.detect_box = None
        
        # Initialize a couple of intermediate image variables
        self.grey = None
        self.small_image = None  
        
        # What kind of detector do we want to load
        self.detector_loaded = False
        self.last_face_box = None
        
        self.hits = 0
        self.misses = 0
        self.hit_rate = 0
        
        # Wait until the image topics are ready before starting
        rospy.wait_for_message("input_rgb_image", Image)
        
        if self.use_depth_for_detection:
            rospy.wait_for_message("input_depth_image", Image)
            
        rospy.loginfo("Ready.")

    def process_image(self, cv_image):
        # Create a greyscale version of the image
        self.grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # STEP 1. Load a detector if one is specified
        if not self.detector_loaded:
            self.detector_loaded = self.load_detector()
            
        # STEP 2: Detect the object
        self.detect_box = self.detect_face(cv_image)
        
        if self.detect_box is not None:
            self.hits += 1
        else:
            self.misses += 1
            
        self.hit_rate = float(self.hits) / (self.hits + self.misses)
                
        return cv_image
    
    def load_detector(self):
        try:
            """ Set up the Haar face detection parameters """
            cascade_frontal_alt = rospy.get_param("~cascade_frontal_alt", "")
            cascade_frontal_alt2 = rospy.get_param("~cascade_frontal_alt2", "")
            cascade_profile = rospy.get_param("~cascade_profile", "")
            cascade_upperbody = rospy.get_param("~cascade_upperbody", "")
            cascade_eye = rospy.get_param("~cascade_eye", "")
            
            self.cascade_frontal_alt = cv2.CascadeClassifier(cascade_frontal_alt)
            self.cascade_frontal_alt2 = cv2.CascadeClassifier(cascade_frontal_alt2)
            self.cascade_profile = cv2.CascadeClassifier(cascade_profile)
            self.cascade_upperbody = cv2.CascadeClassifier(cascade_upperbody)
            self.cascade_eye = cv2.CascadeClassifier(cascade_eye)

            self.min_size = (20, 20)
            self.image_scale = 2
            self.scale_factor = 1.3
            self.min_neighbors = 1
            self.haar_flags = cv.CV_HAAR_DO_CANNY_PRUNING
            
            return True
        except:
            rospy.loginfo("Exception loading face detector!")
            return False

    def detect_face(self, cv_image):
        """ Equalize the histogram to reduce lighting effects. """
        self.grey = cv2.equalizeHist(self.grey)
        
        if self.use_last_face_box and self.last_face_box is not None:
            self.search_scale = 1.5
            x, y, w, h = self.last_face_box
            w_new = int(self.search_scale * w)
            h_new = int(self.search_scale * h)
            search_box = (max(0, int(x - (w_new - w)/2)), max(0, int(y - (h_new - h)/2)), min(self.frame_size[0], w_new), min(self.frame_size[1], h_new))
            sx, sy, sw, sh = search_box
            pt1 = (sx, sy)
            pt2 = (sx + sw, sy + sh)
            search_image = self.grey[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            if self.show_boxes:
                cv2.rectangle(self.marker_image, pt1, pt2, cv.RGB(255, 255, 50), 2)
        else:
            """ Reduce input image size for faster processing """
            search_image = cv2.resize(self.grey, (self.grey.shape[1] / self.image_scale, self.grey.shape[0] / self.image_scale))
                
        """ First check one of the frontal templates """
        if self.cascade_frontal_alt2:
            faces = self.cascade_frontal_alt2.detectMultiScale(search_image, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=self.min_size, flags = self.haar_flags)
                                         
        """ If that fails, check the profile template """
        if not len(faces):
            if self.cascade_profile:
                faces = self.cascade_profile.detectMultiScale(search_image, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=self.min_size, flags = self.haar_flags)

#        """ If that fails, check a different frontal profile """
#        if not len(faces):
#            faces = self.cascade_frontal_alt.detectMultiScale(search_image, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=self.min_size, flags = self.haar_flags)

            if not len(faces):
                self.last_face_box = None
                if self.show_text:
                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(self.marker_image, "LOST FACE!", (20, int(self.frame_size[1] * 0.9)), font_face, font_scale, cv.RGB(255, 50, 50))
                return None
            
        if self.show_text:
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            cv2.putText(self.marker_image, "Hit Rate: " + str(trunc(self.hit_rate, 2)), (20, int(self.frame_size[1] * 0.9)), font_face, font_scale, cv.RGB(255, 255, 0))
                
        for (x, y, w, h) in faces:
            """ The input to the Haar detector was resized, so scale the 
                bounding box of each face and convert it to two CvPoints """
            if self.use_last_face_box and self.last_face_box is not None:
                s_x, s_y, s_w, s_h = search_box
                pt1 = x + s_x, y + s_y
                pt2 = pt1[0] + w, pt1[1] + h
                face_width, face_height = w, h
                self.last_face_box = None
            else:
                pt1 = (int(x * self.image_scale), int(y * self.image_scale))
                pt2 = (int((x + w) * self.image_scale), int((y + h) * self.image_scale))
                face_width = pt2[0] - pt1[0]
                face_height = pt2[1] - pt1[1]

            if self.use_depth_for_detection:
                """ Get the average distance over the face box """
                ave_face_distance = 0
                i = 0
                for x in range(pt1[0], pt2[0]):
                    for y in range(pt1[1], pt2[1]):
                        try:
                            face_distance = cv.Get2D(self.depth_image, y, x)
                            z = face_distance[0]
                        except:
                            continue
                        if isnan(z):
                            continue
                        else:
                            ave_face_distance += z
                            i = i + 1

                """ If we are too close to the Kinect, we will get NaN for distances so just accept the detection. """
                if i == 0:
                    face_size = 0
                
                else:
                    """ Compute the size of the face in meters (average of width and height)
                        The Kinect's FOV is about 57 degrees wide which is, coincidentally, about 1 radian.
                    """
                    ave_face_distance = ave_face_distance / float(i)
                    arc = (self.fov_width * float(face_width) / float(self.frame_size[0]) + self.fov_height * float(face_height) / float(self.frame_size[1])) / 2.0
                    face_size = ave_face_distance * arc
                
                if face_size > self.max_face_size:
                    continue
                
            face_box = (pt1[0], pt1[1], face_width, face_height)
            
            if face_box is not None:
                self.ROI = RegionOfInterest()
                self.ROI.x_offset = min(self.frame_size[0], max(0, pt1[0]))
                self.ROI.y_offset = min(self.frame_size[1], max(0, pt2[0]))
                self.ROI.width = min(self.frame_size[0], face_width)
                self.ROI.height = min(self.frame_size[1], face_height)
                
            self.pubROI.publish(self.ROI)
            
            self.last_face_box = face_box

            """ Break out of the loop after the first face """
            return face_box
        
def trunc(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    slen = len('%.*f' % (n, f))
    return float(str(f)[:slen])
    
def main(args):
      FD = FaceDetector("face_detector")
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print "Shutting down face detector node."
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    