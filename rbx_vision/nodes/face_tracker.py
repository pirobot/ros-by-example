#!/usr/bin/env python

""" face_tracker.py - Version 1.0 2012-02-11

    Combines the OpenCV Haar face detector with Good Features to Track and Lucas-Kanade
    optical flow tracking.
     
"""

import roslib
roslib.load_manifest('rbx_vision')
import rospy
from ros2opencv2 import ROS2OpenCV2
import sys
import cv2
import cv2.cv as cv
from sensor_msgs.msg import Image, RegionOfInterest 
import numpy as np
from time import clock

class FaceTracker(ROS2OpenCV2):
    def __init__(self, node_name):
        ROS2OpenCV2.__init__(self, node_name)
        
        self.node_name = node_name
        
        self.n_faces = rospy.get_param("~n_faces", 1)
        self.show_text = rospy.get_param("~show_text", True)
        self.feature_size = rospy.get_param("~feature_size", 1)
        
        # Good Feature paramters
        self.gf_maxCorners = rospy.get_param("~gf_maxCorners", 200)
        self.gf_qualityLevel = rospy.get_param("~gf_qualityLevel", 0.01)
        self.gf_minDistance = rospy.get_param("~gf_minDistance", 3)
        self.gf_blockSize = rospy.get_param("~gf_blockSize", 3)
        self.gf_useHarrisDetector = rospy.get_param("~gf_useHarrisDetector", False)
        self.gf_k = rospy.get_param("~gf_k", 0.04)
        
        # LK parameters
        self.lk_winSize = rospy.get_param("~lk_winSize", (10, 10))
        self.lk_maxLevel = rospy.get_param("~lk_maxLevel", 2)
        self.lk_criteria = rospy.get_param("~lk_criteria", (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.lk_derivLambda = rospy.get_param("~lk_derivLambda", 0.1)
        
        # Haar face detector parameters
        self.haar_scaleFactor = 1.5
        self.haar_minNeighbors = 1
        self.haar_minSize = (20, 20)
        self.haarFlags = cv.CV_HAAR_DO_CANNY_PRUNING
        self.haar_image_scale = 2
        
        cascade_frontal_alt = rospy.get_param("~cascade_frontal_alt", "")
        cascade_frontal_alt2 = rospy.get_param("~cascade_frontal_alt2", "")
        cascade_profile = rospy.get_param("~cascade_profile", "")
        cascade_eye = rospy.get_param("~cascade_eye", "")

        """ Load the Haar face classifiers """
        self.cascade_frontal_alt = cv2.CascadeClassifier(cascade_frontal_alt)
        self.cascade_frontal_alt2 = cv2.CascadeClassifier(cascade_frontal_alt2)
        self.cascade_profile = cv2.CascadeClassifier(cascade_profile)
        #self.cascade_eye = cv2.CascadeClassifier(cascade_eye)

        self.gf_params = dict( maxCorners = self.gf_maxCorners, 
                       qualityLevel = self.gf_qualityLevel,
                       minDistance = self.gf_minDistance,
                       blockSize = self.gf_blockSize,
                       useHarrisDetector = self.gf_useHarrisDetector,
                       k = self.gf_k )
        
        self.lk_params = dict( winSize  = self.lk_winSize, 
                  maxLevel = self.lk_maxLevel, 
                  criteria = self.lk_criteria,
                  derivLambda = self.lk_derivLambda )
        
        self.haar_params = dict ( scaleFactor = self.haar_scaleFactor,
                                  minNeighbors = self.haar_minNeighbors,
                                  minSize = self.haar_minSize,
                                  flags = self.haarFlags )
        
        self.use_depth_for_detection = rospy.get_param("~use_depth_for_detection", False)
        self.fov_width = rospy.get_param("~fov_width", 1.094)
        self.fov_height = rospy.get_param("~fov_height", 1.094)
        self.max_object_size = rospy.get_param("~max_face_size", 0.28)

        self.detect_interval = 1
        self.keypoints = []

        self.detect_box = None
        self.track_box = None
        
        self.prev_grey = None
        
        self.last_face_box = None
        
        rospy.loginfo("Waiting for video topics to become available...")

        # Wait until the image topics are ready before starting
        rospy.wait_for_message("input_rgb_image", Image)
        
        if self.use_depth_for_detection:
            rospy.wait_for_message("input_depth_image", Image)
            
        rospy.loginfo("Ready.")

    def process_image(self, cv_image):  
        try:              
            # Create a numpy array version of the image as required by many of the cv2 functions
            cv_array = np.array(cv_image, dtype=np.uint8)
    
            # Create a greyscale version of the image
            self.grey = cv2.cvtColor(cv_array, cv2.COLOR_BGR2GRAY)
            
            """ STEP 1: Detect the face if we haven't already """
            if self.detect_box is None:
                self.detect_box = self.detect_face(cv_image)
            
            """ Step 2: Extract keypoints """
            if self.track_box is None or not self.is_rect_nonzero(self.track_box):
                self.track_box = self.detect_box
                self.keypoints = []
                self.get_keypoints(self.track_box)
                
            if self.prev_grey is None:
                self.prev_grey = self.grey
    
            """ Step 3:  Track keypoints using optical flow """
            self.track_box = self.track_keypoints()
            
            # Process any special keyboard commands for this module
            if 32 <= self.keystroke and self.keystroke < 128:
                cc = chr(self.keystroke).lower()
                if cc == 'c':
                    self.keypoints = []
                    self.track_box = None
                    self.detect_box = None
                    
            self.prev_grey = self.grey
        except:
            pass
                
        return cv_image
    
    def detect_face(self, cv_image):
        """ Equalize the histogram to reduce lighting effects. """
        self.grey = cv2.equalizeHist(self.grey)
        
        self.last_face_box = None
        if self.last_face_box is not None:
            self.search_scale = 1.5
            x, y, w, h = self.last_face_box
            w_new = int(self.search_scale * w)
            h_new = int(self.search_scale * h)
            search_box = (max(0, int(x - (w_new - w)/2)), max(0, int(y - (h_new - h)/2)), min(self.frame_size[0], w_new), min(self.frame_size[1], h_new))
            sx, sy, sw, sh = search_box
            pt1 = (sx, sy)
            pt2 = (sx + sw, sy + sh)
            search_image = self.grey[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            cv.Rectangle(self.marker_image, pt1, pt2, cv.RGB(0, 255, 0), 3)
        else:
            """ Reduce input image size for faster processing """
            search_image = cv2.resize(self.grey, (self.grey.shape[1] / self.haar_image_scale, self.grey.shape[0] / self.haar_image_scale))
                
        """ First check one of the frontal templates """
        if self.cascade_frontal_alt2:
            faces = self.cascade_frontal_alt2.detectMultiScale(search_image, **self.haar_params)
                                         
        """ If that fails, check the profile template """
        if not len(faces):
            if self.cascade_profile:
                faces = self.cascade_profile.detectMultiScale(search_image, **self.haar_params)

        """ If that fails, check a different frontal profile """
        if not len(faces):
                if self.cascade_frontal_alt:
                    faces = self.cascade_frontal_alt.detectMultiScale(search_image, **self.haar_params)

        if not len(faces):
            self.last_face_box = None
            if self.show_text:
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv.PutText(self.marker_image, "LOST FACE!", (50, int(self.frame_size[1] * 0.9)), font_face, font_scale, cv.RGB(255, 255, 0))
            return None
                
        for (x, y, w, h) in faces:
            """ The input to cv.HaarDetectObjects was resized, so scale the 
                bounding box of each face and convert it to two CvPoints """
            if self.last_face_box is not None:
                s_x, s_y, s_w, s_h = search_box
                pt1 = x + s_x, y + s_y
                pt2 = pt1[0] + w, pt1[1] + h
                face_width, face_height = w, h
                self.last_face_box = None
            else:
                pt1 = (int(x * self.haar_image_scale), int(y * self.haar_image_scale))
                pt2 = (int((x + w) * self.haar_image_scale), int((y + h) * self.haar_image_scale))
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

    def get_keypoints(self, track_box):
        """ Zero the mask with all black pixels """
        self.mask = np.zeros_like(self.grey)
 
        """ Get the coordinates and dimensions of the track box """
        try:
            x,y,w,h = track_box
        except:
            return None
        
        """ Set the rectangule within the mask to white """
        self.mask[y:y+h, x:x+w] = 255
                
        for x, y in [np.int32(p[-1]) for p in self.keypoints]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(self.grey, mask = self.mask, **self.gf_params)
        
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.keypoints.append((x, y))
                cv2.circle(self.marker_image, (x, y), self.feature_size, (0, 255, 0, 0), cv.CV_FILLED, 8, 0)                
                    
    def track_keypoints(self):
        if len(self.keypoints) > 0:
            img0, img1 = self.prev_grey, self.grey
            p0 = np.float32([p for p in self.keypoints]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
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
        else:
            keypoints_box = None
        
        return keypoints_box
        
    
def main(args):
      FaceTracker("face_tracker")
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print "Shutting down face tracker node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    