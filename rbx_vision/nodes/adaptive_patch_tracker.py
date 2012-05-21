#!/usr/bin/env python

""" adaptive_patch_tracker.py - Version 1.0 2012-02-11

    Combines Good Features to Track and Lucas-Kanade optical flow with adaptive template matching
     
"""

import roslib
roslib.load_manifest('pi_video_tracker')
import rospy
from ros2opencv2 import ROS2OpenCV2
import sys
import cv2
import cv2.cv as cv
from sensor_msgs.msg import Image, RegionOfInterest 
import numpy as np
from time import clock

class AdaptivePatchTracker(ROS2OpenCV2):
    def __init__(self, node_name):
        ROS2OpenCV2.__init__(self, node_name)
        
        self.node_name = node_name
        
        self.n_faces = rospy.get_param("~n_faces", 1)
        self.show_text = rospy.get_param("~show_text", True)
        
        # Good Feature paramters
        self.maxCorners = rospy.get_param("~maxCorners", 200)
        self.qualityLevel = rospy.get_param("~qualityLevel", 0.01)
        self.minDistance = rospy.get_param("~minDistance", 7)
        self.blockSize = rospy.get_param("~blockSize", 10)
        self.useHarrisDetector = rospy.get_param("~useHarrisDetector", True)
        self.k = rospy.get_param("~k", 0.04)
        
        # LK parameters
        self.winSize = rospy.get_param("~winSize", (15, 15))
        self.maxLevel = rospy.get_param("~maxLevel", 2)
        self.criteria = rospy.get_param("~criteria", (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
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
        
        self.use_depth_for_detection = rospy.get_param("~use_depth_for_detection", False)
        self.fov_width = rospy.get_param("~fov_width", 1.094)
        self.fov_height = rospy.get_param("~fov_height", 1.094)
        self.max_object_size = rospy.get_param("~max_face_size", 0.28)

        self.keypoints = []

        self.detect_box = None
        self.track_box = None
        
        self.mask = None
        self.prev_grey = None
        
        self.mask = None
        self.min_template_size = 50
        self.scale_factor = 1.3
        self.template_sum = None
        self.template = None
        self.n_samples = 0
        self.templates = list()
        self.frame_count = 0
        self.detect_interval = 100
        self.acquire_interval = 50
        
        self.classifier_initialized = False
        
        # What kind of detector do we want to load
        self.detector_type = ""
        self.detector_loaded = False
        
        rospy.loginfo("Waiting for video topics to become available...")

        # Wait until the image topics are ready before starting
        rospy.wait_for_message("input_rgb_image", Image)
        
        if self.use_depth_for_detection:
            rospy.wait_for_message("input_depth_image", Image)
            
        rospy.loginfo("Ready.")

    def process_image(self, cv_image):
        self.frame_count = self.frame_count + 1

        # Create a numpy array version of the image as required by many of the cv2 functions
        cv_array = np.array(cv_image, dtype=np.uint8)

        # Create a greyscale version of the image
        self.grey = cv2.cvtColor(cv_array, cv2.COLOR_BGR2GRAY)
        
        # And equalize it
        self.grey = cv2.equalizeHist(self.grey)
        
        # Periodically use the current classifier to re-detect the tracked object
        if self.classifier_initialized:
            if self.frame_count % self.acquire_interval == 0 and len(self.templates) < 5 and self.track_box is not None:
                (template, template_roi) = self.get_template(self.track_box, cv_array)
                if template is not None:
                    self.templates.append(template)
                self.frame_count = 0
                
            elif self.frame_count % self.detect_interval == 0:
                self.keypoints = list()
                self.detect_box = self.redetect_roi(cv_array)
                
                if self.detect_box is not None:
                    self.track_box = self.detect_box
                
                self.frame_count = 0           

        
        """ STEP 1. Load a detector if one is specified """
        if self.detector_type and not self.detector_loaded:
            self.detector_loaded = self.load_detector(self.detector_type)
            
        """ STEP 2: Detect the object """
        if self.detect_box is None:
            """ If we are using a named detector, then run it now  """
            if self.detector_loaded:
                self.detect_box = self.detect_roi(self.detector_type)
            else:
                # Otherwise, wait until the user manually selections an ROI 
                return cv_image         
        else:
            """ STEP 3: Initialize the classifier """
            self.detector_loaded = True
        
            """ Step 4: If we haven't started tracking, initialize the track box to be the detect box and 
                        extract the keypoints within it. """
            if self.track_box is None or not self.is_rect_nonzero(self.track_box):
                self.track_box = self.detect_box
                self.keypoints = []
                self.get_keypoints(self.track_box)
                
            if self.prev_grey is None:
                self.prev_grey = self.grey
    
            """ Step 4:  Now that have keypoints, track them to the next frame using optical flow """
            self.track_box = self.track_keypoints()
            
            # We have an initial ROI, so proceed with intializing the classifier
            if not self.classifier_initialized and self.track_box is not None:
                rospy.loginfo("Initializing classifier...")
                self.classifier_initialized = self.initialize_classifier(self.track_box, cv_array)
        
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
    
    def load_detector(self, detector):
        if detector == "face":
            try:
                """ Set up the Haar face detection parameters """
                self.cascade_frontal_alt = rospy.get_param("~cascade_frontal_alt", "")
                self.cascade_frontal_alt2 = rospy.get_param("~cascade_frontal_alt2", "")
                self.cascade_profile = rospy.get_param("~cascade_profile", "")
                
                self.cascade_frontal_alt = cv2.CascadeClassifier(self.cascade_frontal_alt)
                self.cascade_frontal_alt2 = cv2.CascadeClassifier(self.cascade_frontal_alt2)
                self.cascade_profile = cv2.CascadeClassifier(self.cascade_profile)
        
                self.min_size = (20, 20)
                self.image_scale = 2
                self.haar_scale = 1.5
                self.min_neighbors = 1
                self.haar_flags = cv.CV_HAAR_DO_CANNY_PRUNING
                #self.HaarMemStorage = cv.CreateMemStorage(0)
                
                return True
            except:
                rospy.loginfo("Exception loading face detector!")
                return False
        else:
            return False
        
    def initialize_classifier(self, roi, cv_array):
        (template, template_roi) = self.get_template(roi, cv_array)
        self.templates.append(template)
        rospy.loginfo("Got Template")
        return True
    
    def detect_roi(self, detector):
        if detector == "face":
            detect_box = self.detect_face()
        else:
            detect_box = None
            
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
        try:
            template = cv_array[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    
            cv.NamedWindow("Template", cv.CV_NORMAL)
            cv.ResizeWindow("Template", 320, 240)
            cv.MoveWindow("Template", 700, 50)
            cv.ShowImage("Template", cv.fromarray(template))
            return (template, roi)
        except:
            rospy.loginfo("Exception getting template!")
            return (None, None)
        
    def redetect_roi(self, cv_array):
        detect_box = self.match_template(cv_array)
        return detect_box
        
    def match_template(self, frame):
        self.n_pyr = 2
        
        # Track which scale and rotation gives the best match
        maxScore = -1
        best_s = 1
        best_r = 0
        best_x = 0
        best_y = 0
        
        for template in self.templates:
            if True:
                """ Compute the min and max scales """
                width_ratio = float(self.frame_size[0]) / template.shape[0]
                height_ratio = float(self.frame_size[1]) / template.shape[1]
                
                max_scale = 0.9 * min(width_ratio, height_ratio)
                
                max_template_dimension = max(template.shape[0], template.shape[1])
                min_scale = 1.1 * float(self.min_template_size) / max_template_dimension
                
                scales = list()
                scale = min_scale
                while scale < max_scale:
                    scales.append(scale)
                    scale *= self.scale_factor
                                    
                rotations = [-45, 0, 45]
            else:
                scales = [1]
                rotations = [0]
                        
            H,W = frame.shape[0], frame.shape[1]
            h,w = template.shape[0], template.shape[1]
    
            # Make sure that the template image is smaller than the source
            if W < w or H < h:
                rospy.loginfo( "Template image must be smaller than video frame." )
                return False
            
            if frame.dtype != template.dtype: 
                rospy.loginfo("Template and video frame must have same depth and number of channels.")
                return False
            
            # Create a copy of the frame to modify
            frame_copy = frame.copy()
            
            for i in range(self.n_pyr):
                frame_copy = cv2.pyrDown(frame_copy)
                
            template_height, template_width  = template.shape[:2]
            
            # Cycle through all scales starting with the last successful scale
            #scales = self.scales[self.last_scale:] + self.scales[:self.last_scale - 1]
    
            for s in scales:
                for r in rotations:
                    # Scale the template by s
                    template_copy = cv2.resize(template, (int(template_width * s), int(template_height * s)))
    
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
                        #self.last_scale = self.scales.index(s)
                        best_result = result.copy()
                
        # Transform back to original image sizes
        best_x *= int(pow(2.0, self.n_pyr))
        best_y *= int(pow(2.0, self.n_pyr))
        h,w = template.shape[:2]
        h = int(h * best_s)
        w = int(w * best_s)
        
        best_result = cv2.resize(best_result, (int(pow(2.0, self.n_pyr)) * best_result.shape[1], int(pow(2.0, self.n_pyr)) * best_result.shape[0]))
        cv2.imshow("Result", best_result)
        best_template = cv2.resize(best_template, (int(pow(2.0, self.n_pyr)) * best_template.shape[1], int(pow(2.0, self.n_pyr)) * best_template.shape[0]))
        cv2.imshow("Best Template", best_template)
        
        #match_box = ((best_x + w/2, best_y + h/2), (w, h), -best_r)
        rospy.loginfo("Max Score: " + str(maxScore))
        if maxScore > 0.8:
            return (best_x, best_y, w, h)
        else:
            return None
    
    def redetect_roi1(self, image):
        image_ipl = cv.CreateImage((image.cols, image.rows), cv.IPL_DEPTH_8U, 3)
        cv.Copy(image, image_ipl)

        try:
            #if not self.match_box:
            self.match_box = self.track_box
#            if self.is_rect_nonzero(self.match_box):
#                (center, size, angle) = self.match_box
#                roi_w = int(size[0] * 2)
#                roi_h = int(size[1] * 2)
#                roi_x1 = max(0, int(center[0] - roi_w / 2))
#                roi_y1 = max(0, int(center[1] - roi_h / 2))
#                roi_x2 = min(image_ipl.width, roi_x1 + roi_w)
#                roi_y2 = min(image_ipl.height, roi_y1 + roi_h)
#                roi_w = roi_x2 - roi_x1
#                roi_h = roi_y2 - roi_y1
#            else:
            rospy.loginfo("Scanning whole image...")
            roi_x1 = 0
            roi_y1 = 0
            roi_w = image_ipl.width
            roi_h = image_ipl.height
        except:
            rospy.loginfo("ROI EXCEPTION! Scanning whole image...")
            roi_x1 = 0
            roi_y1 = 0
            roi_w = image_ipl.width
            roi_h = image_ipl.height
            
        roi = (roi_x1, roi_y1, roi_w, roi_h)
        cv.SetImageROI(image_ipl, roi)

        roi_corners = list()
        
        index = -1
        for template_ipl in self.templates:
            index += 1
            
            W,H = cv.GetSize(image_ipl)
            w,h = cv.GetSize(template_ipl)
            width = W - w + 1
            height = H - h + 1
            if width <= 0 or height <= 0:
                continue
    
            try:
                result = cv.CreateImage((width, height), 32, 1)
                #tmp = cv.CreateImage((width, height), 32, 1)
                cv.MatchTemplate(image_ipl, template_ipl, result, cv.CV_TM_CCOEFF_NORMED)
                (min_score, max_score, minloc, maxloc) = cv.MinMaxLoc(result)

                #cv.ShowImage("Current Template ", template_ipl)
                #cv.ShowImage("Current Image ", image_ipl)
                #cv.SetZero(tmp)
                #cv.AddS(tmp, 1.0, tmp)
                #cv.Sub(tmp, result, result)
                cv.NamedWindow("Template Matching Result", cv.CV_NORMAL)
                cv.ResizeWindow("Template Matching Result", 320, 240)
                cv.MoveWindow("Template Matching Result", 700, 350)
                cv.ShowImage("Template Matching Result", result)
                
                rospy.loginfo(str(index) + ": " + str(min_score))
                
                # If score falls above threshold, reject the template
                #if min_score > 0.35:
                   #return None
                
                (x, y) = maxloc
                corrected_x = int(x) + roi_x1
                corrected_y = int(y) + roi_y1
        
                match_box = (corrected_x, corrected_y, w, h)
                corner1, corner2 = (corrected_x, corrected_y), (corrected_x + w, corrected_y + h)
                roi_corners.append(corner1)
                roi_corners.append(corner2)
                cv.Rectangle(self.marker_image, corner1, corner2, (0, 255, 0), 3, 0)
            except:
                rospy.loginfo("EXCEPTION!")
                continue
        # Now find the smallest rectangle that encompasses all matching templates
        cv.ResetImageROI(image_ipl)
        new_roi = cv.BoundingRect(roi_corners)
        corner1, corner2 = (new_roi[0], new_roi[1]), (int(new_roi[0] + new_roi[2] / 2), int(new_roi[1] + new_roi[3] / 2))
        cv.Rectangle(self.marker_image, corner1, corner2, (0, 255, 0), 3, 0)
        return new_roi
    
    def detect_face(self):        
        """ Equalize the histogram to reduce lighting effects. """
        self.grey = cv2.equalizeHist(self.grey)
    
        """ Scale input image for faster processing """
        small_image = cv2.resize(self.grey, (self.grey.shape[1] / self.image_scale, self.grey.shape[0] / self.image_scale))
            
        """ First check one of the frontal templates """
        if self.cascade_frontal_alt:
            #faces = cv.HaarDetectObjects(small_image, self.cascade_frontal_alt, cv.CreateMemStorage(0),
            #self.haar_scale, self.min_neighbors, self.haar_flags, self.min_size)
            faces = self.cascade_frontal_alt.detectMultiScale(small_image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
                                         
        """ If that fails, check the profile template """
        if not len(faces):
            if self.cascade_profile:
                faces = self.cascade_profile.detectMultiScale(small_image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)

            if not len(faces):
                """ If that fails, check a different frontal profile """
                faces = self.cascade_frontal_alt2.detectMultiScale(small_image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)

        if not len(faces):
            if self.show_text:
                hscale = 0.4 * self.frame_size[0] / 160. + 0.1
                vscale = 0.4 * self.frame_size[1] / 120. + 0.1
                text_font = cv.InitFont(cv.CV_FONT_VECTOR0, hscale, vscale, 0, 1, 8)
                cv.PutText(self.marker_image, "LOST FACE!", (50, int(self.frame_size[1] * 0.9)), text_font, cv.RGB(255, 255, 0))
            return None
                
        for (x, y, w, h) in faces:
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

        """ For manually selected regions, just use a rectangle """
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        
        """ Set the rectangule within the mask to white """
        self.mask[y:y+h, x:x+w] = 255
                
        for x, y in [np.int32(tr[-1]) for tr in self.keypoints]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        p = cv2.goodFeaturesToTrack(self.grey, mask = self.mask, **self.gf_params)
        
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.keypoints.append([(x, y)])
                cv.Circle(self.marker_image, (x, y), 3, (0, 255, 0, 0), cv.CV_FILLED, 8, 0)                
                    
    def track_keypoints(self):
        if len(self.keypoints) > 0:
            img0, img1 = self.prev_grey, self.grey
            p0 = self.keypoints
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
                cv.Circle(self.marker_image, (x, y), 3, (0, 255, 0, 0), cv.CV_FILLED, 8, 0)
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
      APT = AdaptivePatchTracker("adaptive_patch_tracker")
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print "Shutting down adaptive patch tracker node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    