#!/usr/bin/env python

""" object_tracker.py - Version 1.0 2012-02-11

"""

import roslib
roslib.load_manifest('pi_video_tracker')
import rospy
import sys
import cv2
import cv as cv1
from sensor_msgs.msg import Image, RegionOfInterest

class ObjectTracker:
    def __init__(self, node_name):
                
        self.node_name = node_name
        
        image_stream = self.subscribe(self.image_topic)
        depth_stream = self.subscribe(self.depth_topic)

        self.use_depth_for_detection = rospy.get_param("~use_depth_for_detection", False)
        self.fov_width = rospy.get_param("~fov_width", 1.094)
        self.fov_height = rospy.get_param("~fov_height", 1.094)
        self.max_face_size = rospy.get_param("~max_face_size", 0.28)
        
        # Intialize the detection box
        self.detect_box = None
        
        # Initialize a couple of intermediate image variables
        self.grey = None
        self.small_image = None  
        
        # What kind of detector do we want to load
        self.detector_type = "face"
        self.detector_loaded = False
        
        rospy.loginfo("Waiting for video topics to become available...")

        # Wait until the image topics are ready before starting
        rospy.wait_for_message("input_rgb_image", Image)
        
        if self.use_depth_for_detection:
            rospy.wait_for_message("input_depth_image", Image)
            
        rospy.loginfo("Ready.")

    def process_image(self, cv_image):        
        # STEP 1. Load a detector if one is specified
        if self.detector_type and not self.detector_loaded:
            self.detector_loaded = self.load_detector(self.detector_type)
            
        # STEP 2: Detect the object
        self.detect_box = self.detect_roi(self.detector_type, cv_image)
                
        return cv_image
    
    def load_detector(self, detector):
        if detector == "face":
            try:
                """ Set up the Haar face detection parameters """
                self.cascade_frontal_alt = rospy.get_param("~cascade_frontal_alt", "")
                self.cascade_frontal_alt2 = rospy.get_param("~cascade_frontal_alt2", "")
                self.cascade_profile = rospy.get_param("~cascade_profile", "")
                
                self.cascade_frontal_alt = cv1.Load(self.cascade_frontal_alt)
                self.cascade_frontal_alt2 = cv1.Load(self.cascade_frontal_alt2)
                self.cascade_profile = cv1.Load(self.cascade_profile)
        
                self.min_size = (20, 20)
                self.image_scale = 2
                self.haar_scale = 1.5
                self.min_neighbors = 1
                self.haar_flags = cv1.CV_HAAR_DO_CANNY_PRUNING
                                
                return True
            except:
                rospy.loginfo("Exception loading face detector!")
                return False
        else:
            return False
        
    def detect_roi(self, detector, cv_image):
        if detector == "face":
            detect_box = self.detect_face(cv_image)
        
        return detect_box
    
    def detect_face(self, cv_image):
        if not self.grey:
            self.grey = cv1.CreateImage(cv1.GetSize(self.frame), 8, 1)
            
        if not self.small_image:
            self.small_image = cv1.CreateImage((cv1.Round(self.frame_size[0] / self.image_scale),
                       cv1.Round(self.frame_size[1] / self.image_scale)), 8, 1)
    
        """ Convert color input image to grayscale """
        cv1.CvtColor(cv_image, self.grey, cv1.CV_BGR2GRAY)
        
        """ Equalize the histogram to reduce lighting effects. """
        cv1.EqualizeHist(self.grey, self.grey)
    
        """ Scale input image for faster processing """
        cv1.Resize(self.grey, self.small_image, cv1.CV_INTER_LINEAR)
    
        """ First check one of the frontal templates """
        if self.cascade_frontal_alt:
            faces = cv1.HaarDetectObjects(self.small_image, self.cascade_frontal_alt, cv1.CreateMemStorage(0),
                                          self.haar_scale, self.min_neighbors, self.haar_flags, self.min_size)
                                         
        """ If that fails, check the profile template """
        if not faces:
            if self.cascade_profile:
                faces = cv1.HaarDetectObjects(self.small_image, self.cascade_profile, cv1.CreateMemStorage(0),
                                             self.haar_scale, self.min_neighbors, self.haar_flags, self.min_size)

            if not faces:
                """ If that fails, check a different frontal profile """
                if self.cascade_frontal_alt2:
                    faces = cv1.HaarDetectObjects(self.small_image, self.cascade_frontal_alt2, cv1.CreateMemStorage(0),
                                         self.haar_scale, self.min_neighbors, self.haar_flags, self.min_size)
            
        if not faces:
            if self.show_text:
                hscale = 0.4 * self.frame_size[0] / 160. + 0.1
                vscale = 0.4 * self.frame_size[1] / 120. + 0.1
                text_font = cv1.InitFont(cv1.CV_FONT_VECTOR0, hscale, vscale, 0, 2, 8)
                if self.frame_size[0] >= 640:
                    vstart = 400
                    voffset = int(50 + self.frame_size[1] / 120.)
                elif self.frame_size[0] == 320:
                    vstart = 200
                    voffset = int(35 + self.frame_size[1] / 120.)
                else:
                    vstart = 100
                cv1.PutText(self.marker_image, "LOST FACE!", (10, vstart), text_font, cv1.RGB(255, 255, 0))
            return None
                
        for ((x, y, w, h), n) in faces:
            """ The input to cv.HaarDetectObjects was resized, so scale the 
                bounding box of each face and convert it to two CvPoints """
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
                            face_distance = cv1.Get2D(self.depth_image, y, x)
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
            
            if face_box:
                self.ROI = RegionOfInterest()
                self.ROI.x_offset = min(self.frame_size[0], max(0, pt1[0]))
                self.ROI.y_offset = min(self.frame_size[1], max(0, pt2[0]))
                self.ROI.width = min(self.frame_size[0], face_width)
                self.ROI.height = min(self.frame_size[1], face_height)
                
            self.pubROI.publish(self.ROI)

            """ Break out of the loop after the first face """
            return face_box
    
def main(args):
      FD = FaceDetector("face_detector")
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print "Shutting down face detector node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    