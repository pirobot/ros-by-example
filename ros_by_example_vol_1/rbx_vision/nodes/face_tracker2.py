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
        super(FaceTracker, self).__init__(node_name)

        self.n_faces = rospy.get_param("~n_faces", 1)
        self.show_text = rospy.get_param("~show_text", True)
        self.feature_size = rospy.get_param("~feature_size", 1)
        
        self.use_depth_for_tracking = rospy.get_param("~use_depth_for_tracking", False)
        self.auto_min_keypoints = rospy.get_param("~auto_min_keypoints", True)
        self.min_keypoints = rospy.get_param("~min_keypoints", 50) # Used only if auto_min_keypoints is False
        self.abs_min_keypoints = rospy.get_param("~abs_min_keypoints", 6)
        self.std_err_xy = rospy.get_param("~std_err_xy", 3.0) 
        self.pct_err_z = rospy.get_param("~pct_err_z", 0.42) 
        self.max_mse = rospy.get_param("~max_mse", 20000)
        self.keypoint_type = rospy.get_param("~keypoint_type", 0)
        self.add_keypoint_distance = rospy.get_param("~add_keypoint_distance", 10)
        self.add_keypoints_interval = rospy.get_param("~add_keypoints_interval", 1)
        self.drop_keypoints_interval = rospy.get_param("~drop_keypoints_interval", 1)
        self.expand_roi_init = rospy.get_param("~expand_roi", 1.02)
        self.expand_roi = self.expand_roi_init

        self.frame_index = 0
        self.add_index = 0
        self.drop_index = 0
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
        if self.detect_box is not None:
            if not self.track_box or not self.is_rect_nonzero(self.track_box):
                self.track_box = self.detect_box
                self.keypoints = list()
                self.keypoints = self.get_keypoints(self.grey, self.track_box)
                  
            if self.prev_grey is None:
                self.prev_grey = self.grey           
              
            """ Step 3:  Begin tracking """
            self.track_box = self.track_keypoints(self.grey, self.prev_grey)
          
            """ STEP 4: Drop keypoints that are too far from the main cluster """
            if self.frame_index % self.drop_keypoints_interval == 0 and len(self.keypoints) > 0:
                ((cog_x, cog_y, cog_z), mse_xy, mse_z, score) = self.drop_keypoints(self.abs_min_keypoints, self.std_err_xy, self.max_mse)
                
                if score == -1:
                    self.detect_box = None
                    self.track_box = None
                    return cv_image
              
            """ STEP 5: Add keypoints if the number is getting too low """
            if self.frame_index % self.add_keypoints_interval == 0 and len(self.keypoints) < self.min_keypoints:
                self.expand_roi = self.expand_roi_init * self.expand_roi
                self.add_keypoints(self.track_box)
            else:
                self.frame_index += 1
                self.expand_roi = self.expand_roi_init
                       
        else:         
            self.keypoints = list()
            self.track_box = None
            
        self.prev_grey = self.grey
          
        # Process any special keyboard commands for this module
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 'c':
                self.keypoints = []
                self.track_box = None
                self.detect_box = None
            elif cc == 'a':
                self.auto_face_tracking = not self.auto_face_tracking
        
        return cv_image
    
    def add_keypoints(self, track_box):
        """ Look for any new keypoints around the current feature cloud """
        
        """ Begin with a mask of all black pixels """
        mask = np.zeros_like(self.grey)
        
        """ Get the coordinates and dimensions of the current track box """
        try:
            ((x,y), (w,h), a) = track_box
        except:
            try:
                x,y,w,h = track_box
            except:
                rospy.loginfo("Track box has shrunk to zero...")
                return
        
        x = int(x)
        y = int(y)
        
        """ Expand the track box to look for new keypoints """
        w_new = int(self.expand_roi * w)
        h_new = int(self.expand_roi * h)
        
        pt1 = (x - int(w_new / 2), y - int(h_new / 2))
        pt2 = (x + int(w_new / 2), y + int(h_new / 2))

        cv2.rectangle(self.marker_image, pt1, pt2, cv.RGB(255, 255, 0))
        
        mask_box = ((x, y), (w_new, h_new), a)

        # Rectangular mask
        #mask[y:y+h_new,x:x+w_new] = 255
                        
        """ Create a filled white ellipse within the track_box to define the ROI. """
        cv2.ellipse(mask, mask_box, cv.CV_RGB(255,255, 255), cv.CV_FILLED)
        
        if self.keypoints is not None:
            # Mask the current keypoints
            for x, y in [np.int32(p) for p in self.keypoints]:
                cv2.circle(mask, (x, y), 5, 0, -1)
         
        if self.keypoint_type == 0:
            """ Get the new keypoints using Good Features to Track """
            new_keypoints = cv2.goodFeaturesToTrack(self.grey, mask = mask, **self.gf_params)

        elif self.keypoint_type == 1:
            """ Get the new keypoints using SURF """
            new_keypoints = []
            surf_keypoints, surf_descriptors = self.surf.detect(self.grey, self.mask, False)
            for keypoint in surf_keypoints:
                new_keypoints.append((int(keypoint.pt[0]), int(keypoint.pt[1])))

        """ Append new keypoints to the current list if they are not too far from the current cluster """          
        if new_keypoints is not None:
            for x, y in np.float32(new_keypoints).reshape(-1, 2):
                distance = self.distance_to_cluster((x,y), self.keypoints)
                if distance > self.add_keypoint_distance:
                    self.keypoints.append((x,y))
                    cv2.circle(self.marker_image, (x, y), 3, (255, 255, 0, 0), cv.CV_FILLED, 2, 0)
                                    
            """ Remove duplicate keypoints """
            self.keypoints = list(set(self.keypoints))
        
    def distance_to_cluster(self, test_point, cluster):
        min_distance = 10000
        for point in cluster:
            if point == test_point:
                continue
            """ Use L1 distance since it is faster than L2 """
            distance = abs(test_point[0] - point[0])  + abs(test_point[1] - point[1])
            if distance < min_distance:
                min_distance = distance
        return min_distance
    
    def drop_keypoints(self, min_keypoints, outlier_threshold, mse_threshold):
        sum_x = 0
        sum_y = 0
        sum_z = 0
        sse = 0
        keypoints_xy = self.keypoints
        keypoints_z = self.keypoints
        n_xy = len(self.keypoints)
        n_z = n_xy
        
        if self.use_depth_for_tracking:
            if self.depth_image is None:
                return ((0, 0, 0), 0, 0, -1)
        
        """ If there are no keypoints left to track, start over """
        if n_xy == 0:
            return ((0, 0, 0), 0, 0, -1)
        
        """ Compute the COG (center of gravity) of the cluster """
        for point in self.keypoints:
            sum_x = sum_x + point[0]
            sum_y = sum_y + point[1]
        
        mean_x = sum_x / n_xy
        mean_y = sum_y / n_xy
        
        if self.use_depth_for_tracking:
            for point in self.keypoints:
                try:
                    z = cv.Get2D(self.depth_image, min(self.frame_height - 1, int(point[1])), min(self.frame_width - 1, int(point[0])))
                except:
                    continue
                z = z[0]
                """ Depth values can be NaN which should be ignored """
                if isnan(z):
                    continue
                else:
                    sum_z = sum_z + z
                    
            mean_z = sum_z / n_z
            
        else:
            mean_z = -1
        
        """ Compute the x-y MSE (mean squared error) of the cluster in the camera plane """
        for point in self.keypoints:
            sse = sse + (point[0] - mean_x) * (point[0] - mean_x) + (point[1] - mean_y) * (point[1] - mean_y)
            #sse = sse + abs((point[0] - mean_x)) + abs((point[1] - mean_y))
        
        """ Get the average over the number of feature points """
        mse_xy = sse / n_xy
        
        """ The MSE must be > 0 for any sensible feature cluster """
        if mse_xy == 0 or mse_xy > mse_threshold:
            return ((0, 0, 0), 0, 0, -1)
        
        """ Throw away the outliers based on the x-y variance """
        max_err = 0
        for point in self.keypoints:
            std_err = ((point[0] - mean_x) * (point[0] - mean_x) + (point[1] - mean_y) * (point[1] - mean_y)) / mse_xy
            if std_err > max_err:
                max_err = std_err
            if std_err > outlier_threshold:
                keypoints_xy.remove(point)
                # Briefly mark the removed points in red
                cv2.circle(self.marker_image, (point[0], point[1]), 2, (0, 0, 255), cv.CV_FILLED)   
                try:
                    keypoints_z.remove(point)
                    n_z = n_z - 1
                except:
                    pass
                
                n_xy = n_xy - 1
                                
        """ Now do the same for depth """
        if self.use_depth_for_tracking:
            sse = 0
            for point in keypoints_z:
                try:
                    z = cv.Get2D(self.depth_image, min(self.frame_height - 1, int(point[1])), min(self.frame_width - 1, int(point[0])))
                    z = z[0]
                    sse = sse + (z - mean_z) * (z - mean_z)
                except:
                    n_z = n_z - 1
            
            if n_z != 0:
                mse_z = sse / n_z
            else:
                mse_z = 0
            
            """ Throw away the outliers based on depth using percent error rather than standard error since depth
                 values can jump dramatically at object boundaries  """
            for point in keypoints_z:
                try:
                    z = cv.Get2D(self.depth_image, min(self.frame_height - 1, int(point[1])), min(self.frame_width - 1, int(point[0])))
                    z = z[0]
                except:
                    continue
                try:
                    pct_err = abs(z - mean_z) / mean_z
                    if pct_err > self.pct_err_z:
                        keypoints_xy.remove(point)
                except:
                    pass
        else:
            mse_z = -1
        
        self.keypoints = keypoints_xy
               
        """ Consider a cluster bad if we have fewer than min_keypoints left """
        if len(self.keypoints) < min_keypoints:
            score = -1
        else:
            score = 1

        return ((mean_x, mean_y, mean_z), mse_xy, mse_z, score)
    
if __name__ == '__main__':
    try:
        node_name = "face_tracker"
        FaceTracker(node_name)
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down face tracker node."
        cv.DestroyAllWindows()
