#!/usr/bin/env python

""" adaptive_video_tracker.py - Version 0.1 2011-06-19

    Track a designated region-of-interest in a video using OpenCV's Lucas-Kanade Optical Flow filter
    while also learning a custom classifier for the tracked region.  Based on the OpenTLD project at
    https://github.com/zk00006/OpenTLD/wiki.

    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2011 Patrick Goebel.  All rights reserved.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.5
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
      
"""

import roslib
roslib.load_manifest('pi_video_tracker')
from ros2opencv import ROS2OpenCV
import rospy
import cv2.cv as cv
import cv2
import sys
import os
from sensor_msgs.msg import RegionOfInterest, Image
from math import sqrt, isnan, sin, cos
import time, random
import numpy as np

class VideoTracker(ROS2OpenCV):
    def __init__(self, node_name):
        ROS2OpenCV.__init__(self, node_name)
        
        self.node_name = node_name
        
        # Learning parameters
        self.n_warps = 10
        self.max_overlap = 0.0
        self.shift = 1.0
        self.noise_var = 10
        self.waitkey = 1
        
        self.positive_samples = list()
        self.negative_samples = list()
        self.classes = list()
        
        self.auto_face_tracking = rospy.get_param("~auto_face_tracking", False)
        self.auto_motion_tracking = rospy.get_param("~auto_motion_tracking", False)
        self.show_text = rospy.get_param("~show_text", True)
        self.feature_size = rospy.get_param("~feature_size", 1)
        self.use_haar_only = rospy.get_param("~use_haar_only", False)
        self.use_depth_for_detection = rospy.get_param("~use_depth_for_detection", False)
        self.fov_width = rospy.get_param("~fov_width", 1.094)
        self.fov_height = rospy.get_param("~fov_height", 1.094)
        self.max_face_size = rospy.get_param("~max_face_size", 0.28)
        self.use_depth_for_tracking = rospy.get_param("~use_depth_for_tracking", False)
        self.auto_min_features = rospy.get_param("~auto_min_features", True)
        self.min_features = rospy.get_param("~min_features", 50) # Used only if auto_min_features is False
        self.abs_min_features = rospy.get_param("~abs_min_features", 6)
        self.std_err_xy = rospy.get_param("~std_err_xy", 2.5) 
        self.pct_err_z = rospy.get_param("~pct_err_z", 0.42) 
        self.max_mse = rospy.get_param("~max_mse", 10000)
        self.good_feature_distance = rospy.get_param("~good_feature_distance", 5)
        self.add_feature_distance = rospy.get_param("~add_feature_distance", 10)
        self.flip_image = rospy.get_param("~flip_image", False)
        self.feature_type = rospy.get_param("~feature_type", 0) # 0 = Good Features to Track, 1 = SURF
        self.get_surf_also = False
        self.expand_roi_init = rospy.get_param("~expand_roi", 1.02)
        self.expand_roi = self.expand_roi_init
        
        self.detect_box = None
        self.track_box = None
        self.match_box = None
        self.features = []

        self.mask = None
        self.template_sum = None
        self.template = None
        self.n_samples = 0
        self.image_ipl = None
        self.template_ipl = None
        self.templates = list()
        
        # Initialize a couple of intermediate image variables
        self.grey = None
        self.pyramid = None
        self.small_image = None
        
        # Set the Good Features to Track and Lucas-Kanade parameters
        self.night_mode = False       
        self.quality = 0.01
        self.win_size = 10
        self.max_count = 200
        self.flags = 0
        self.frame_count = 0
        
        # Set the SURF parameters """
        self.surf_hessian_quality = rospy.get_param("~surf_hessian_quality", 300)
        #self.SURFMemStorage = cv.CreateMemStorage(0)
        
        # Classifier parameters """
        self.test_classifier = rospy.get_param("~test_classifier", False)
        self.keypoints = []
        self.descriptors = []
        self.classifier_keypoints = []
        self.classifier_descriptors = []
        
        # A window to show the current template image
#        cv.NamedWindow("Template", cv.CV_NORMAL)
#        cv.ResizeWindow("Template", 160, 120)
#        cv.MoveWindow("Template", 700, 50)
        
        self.detector_loaded = False
        self.classifier_initialized = False
        self.tracking_score_threshold = 0.5
        
        if self.auto_face_tracking:
            self.detector_type = "face"
        else:
            self.detector_type = None
        
        rospy.loginfo("Waiting for video topics...")

        # Wait until the image topics are ready before starting
        rospy.wait_for_message("input_rgb_image", Image)
        
        if self.use_depth_for_detection or self.use_depth_for_tracking:
            rospy.wait_for_message("input_depth_image", Image)
            
        rospy.loginfo("Ready.")

 
    def process_image(self, cv_image):
        """
        Components:
            * load_detector()
            * detect_roi()
            * init_classifier()
            * track_roi()
            * update_classifier()
            * evaluate_roi()
            * detect_roi()
            * save_detector()
            
        After the initial ROI is determined, each frame runs through the following process:
            1. track_roi()
            2. update_classifier()
            3. evaluate()
            4. if evaluate < threshold then detect_roi() using current classifier
            
        These can be broken down as follows:
            0. Detector
                0.1 The initial detection using a stored classifier or other method such as user drawn bounding box, motion, etc.
                0.2 Subsequent detections as needed using online classifier
            1. Tracker
                1.1 Lucas-Kanade Optical Flow tracker of current feature points including NCC and Forward-Backward error checking
            2. Learner
                2.1 Descriptors are extracted for each frame of the tracked ROI
                2.2 These descriptors are tested against the current classifier
                    2.2.1 If correctly classified (positive), the classifier does not need to be updated
                    2.2.2 If classified as negative, classifier is updated with positive example
                2.3 A random sample of far away ROIs is also tested against the classifier
                    2.3.1 If correctly classified (negative), the classifier does not need to be updated
                    2.3.2 If classified as positive, classifier is updated with negative example
            3. Classifier
                3.1 At periodic intervals, or if the confidence in the tracker falls below a threshold, rescan the image
                    using the current classifier to re-locate the target
        """
        #self.busy = True
        
        # Keep track of the number of frames processed.
        self.frame_count = self.frame_count + 1
        
        # Create a mask image to be used in various functions below
        if not self.mask:
            self.mask = cv.CreateImage(cv.GetSize(self.frame), 8, 1)
            
        # Create a greyscale version of the image to be used in various functions below
        if not self.grey:
            self.grey = cv.CreateImage(cv.GetSize(self.frame), 8, 1)
        
        # Periodically use the current classifier to re-detect the tracked object
        if self.classifier_initialized:
            if self.frame_count % 50 == 0:
                self.frame_count = 0
                #self.features = []
                #self.get_features(self.redetect_roi(cv_image))
                #self.features = []
                #if self.detect_box is None or self.track_box is None:
                self.detect_box = self.redetect_roi(cv_image)
                #self.track_box = self.detect_box
                self.get_features(self.detect_box)
                
#                self.match_box = self.match_template(cv_image)
#                self.get_template(self.match_box)
#                if self.match_box and self.is_rect_nonzero(self.match_box):
#                    self.get_features(self.match_box)
#                    self.track_box = self.cvRect_to_cvBox2D(self.match_box)

#        if self.classifier_initialized and self.track_box:
#            roi_ok = self.evaluate_roi(self.track_box)
#            if not roi_ok:
#                self.update_classifier(self.track_box)
        
        ''' STEP 1. Load a detector if one is specified '''
        if self.detector_type and not self.detector_loaded:
            self.detector_loaded = self.load_detector(self.detector_type)
            
        ''' STEP 2: Get the initial ROI '''
        if not self.detect_box:
            """ If we are using a named detector, then run it now  """
            if self.detector_loaded:
                self.detect_box = self.detect_roi(self.detector_type)
            else:
                # Otherwise, wait until the user manually selections an ROI 
                #self.busy = False
                return cv_image         
        else:
            # We have an initial ROI, so proceed with intializing the classifier
            if not self.classifier_initialized:
                rospy.loginfo("Initializing classifier...")
                ''' STEP 3: Initialize the online classifier '''
                self.classifier_initialized = self.initialize_classifier(self.detect_box)
            
            ''' STEP 4: Extract features and their locations from within the ROI '''
            if not self.track_box or not self.is_rect_nonzero(self.track_box):
                self.features = []
                self.track_box = self.detect_box
                self.get_features(self.track_box)
                
            ''' STEP 4a: Prune features that are too far from the main cluster '''
            if len(self.features) > 0:
                ((cog_x, cog_y, cog_z), mse_xy, mse_z, score) = self.prune_features(min_features = self.abs_min_features, outlier_threshold = self.std_err_xy, mse_threshold=self.max_mse)
                
                if score == -1:
                    self.detect_box = None
                    self.track_box = None
                    #self.busy = False
                    return cv_image
            
            ''' STEP 4b: Add features if the number is getting too low '''
            if len(self.features) < self.min_features:
                self.expand_roi = self.expand_roi_init * self.expand_roi
                self.add_features()       
            else:
                self.expand_roi = self.expand_roi_init 
            
            ''' STEP 5: Track the ROI to the next frame '''
            self.track_box = self.track_roi()
            
            ''' STEP 6: Update the classifier with new samples '''
            #self.update_classifier()
            
            ''' STEP 7: Evaluate the quality of the current track box '''
            #self.tracking_score = self.evaluate_tracking()
            
            ''' STEP 8: If tracking quality has fallen below some threshold, re-detect the ROI '''
            #if self.tracking_score < self.tracking_score_threshold:
                #self.detect_box = self.detect_roi()

        # Process any special keyboard commands for this module
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 'a':
                self.auto_face_tracking = not self.auto_face_tracking
                if self.auto_face_tracking:
                    self.features = []
                    self.track_box = None
                    self.detect_box = None
            if cc == 'c':
                self.features = []
                self.track_box = None
                self.detect_box = None
                self.classifier_initialized = True
            elif cc == 'm':
                #self.detect_box = None
                rospy.loginfo("Matching template")
                self.features = []
                self.get_features(self.redetect_roi(cv_image))
                #self.detect_box = self.match_template(cv_image)
                #self.get_features(self.match_box)
                #self.track_box = self.cvRect_to_cvBox2D(self.match_box)         
            elif cc == 'd':
                #self.track_box = None
                self.detect_box = self.detect_face()
            elif cc == 'x':
                self.n_samples = 0
                self.match_box = None
                self.detect_box = None
                self.template = None
                self.template_sum = None
            elif cc == 's':
                if self.track_box is None:
                    (template, self.match_box) = self.get_template(self.detect_box)
                else:
                    (template, self.match_box) = self.get_template(self.track_box)
                
                if self.n_samples == 1:
                    self.classifier_keypoints, self.classifier_descriptors = self.get_descriptors(self.match_box) 
                else:
                    self.keypoints, self.descriptors = self.get_descriptors(self.match_box)
                    self.classify_patch()
                    
        #self.busy = False
                
        return cv_image
    
#    def initialize_classifier(self):
#        try:
#            pos_ex = self.generate_postive_examples(self.detect_box, self.frame)
#            neg_ex = self.generate_negative_examples(self.detect_box, self.frame)
#            
#            train_ex, valid_ex = split()
#            
#            self.classifier = svm()
#            self.classifier.train()
#            self.classifier.validate()
#            return True
#        except:
#            rospy.loginfo("Classifier Initialization Failed")
#            return False

    def update_classifier(self, roi):
        (template, template_roi) = self.get_template(roi)
        if template is not None:
            self.templates.append(template)
        
    def initialize_classifier(self, roi):
        (template, template_roi) = self.get_template(roi)
        self.templates.append(template)
        rospy.loginfo("Got Template")
        return True
    
        # Generate positive samples using the detected ROI
        self.generate_positive_samples(roi, reset=True)
        
        # Generate negative samples using the rest of the image outside the ROI        
        self.generate_negative_samples(roi, reset=True)
        
        return True
             
        sample_data = []
        sample_data.extend(self.positive_samples)
        sample_data.extend(self.negative_samples)
        
        samples = np.array(sample_data, dtype=np.float32)
        responses = np.array(self.classes, dtype=np.float32)
        
        rnd_state = np.random.get_state()
        np.random.shuffle(samples)
        np.random.set_state(rnd_state)
        np.random.shuffle(responses)
        
        model = SVM()
        #model = KNearest()
        #model = RTrees()
        
        train_ratio = 0.7
        train_n = int(len(samples)*train_ratio)
        
        start = time.time()
        model.train(samples[:train_n], responses[:train_n])
        duration = time.time() - start
        
        print "Duration:", duration
        
        train_rate = np.mean(model.predict(samples[:train_n]) == responses[:train_n])                                                                       
        test_rate  = np.mean(model.predict(samples[train_n:]) == responses[train_n:])
        
        print 'train rate: %f  test rate: %f' % (train_rate*100, test_rate*100)
        
        return True
    
    def generate_samples(self, roi, positive=True, reset=True):
        if reset:
            if positive:
                self.positive_samples = []
                # TODO delete class values
            else:
                self.negative_samples = []
                # TODO delete class values

        warp = cv.CreateMat(2, 3, cv.CV_32FC1)    
        ran = random.Random()
        sample = cv.GetSubRect(self.frame, roi)
        
#        TODO: Create mask by filling inward from outer contour.
#        sz = (sample.width & -2, sample.height & -2)
#        sz_sample = cv.CreateMat(sz[1], sz[0], cv.CV_8UC3)
#        cv.Resize(sample, sz_sample)
#        pyr = cv.CreateMat(sz[1]/2, sz[0]/2, cv.CV_8UC3)
        
        #cv.ShowImage("Sample", sample)
        
        noise = cv.CloneMat(sample)
        grey = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)

#        cv.PyrDown(sample, pyr, 7)
#        cv.PyrUp(pyr, sample, 7)
        
        cv.Zero(noise)
        cv.RandArr(cv.RNG(), noise, cv.CV_RAND_NORMAL, (0, 0, 0), (self.noise_var, self.noise_var, self.noise_var))
    
        cv.Add(sample, noise, sample)
        cv.CvtColor(sample, grey, cv.CV_RGB2GRAY)
        
        cv.EqualizeHist(grey, grey)

#        cv.FloodFill(grey, (grey.width/2, grey.height/2), 200, lo_diff=(0, 0, 0, 0), up_diff=(100, 100, 100, 100), flags=4, mask=None)
#        edges = cv.CloneMat(grey)
#        cv.Canny(grey, edges, 0, 200)
#        cv.Dilate(grey, grey, None, 1)
#        
#        contour = cv.FindContours(edges, cv.CreateMemStorage(0), mode=cv.CV_RETR_EXTERNAL)
#        
#        cv.DrawContours(edges, contour, cv.RGB(255, 255, 255), cv.RGB(255, 255, 255), 1, 5)
#        cv.ShowImage("Edges", edges)

        result = cv.CloneMat(grey)

        mask = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)
        warped_mask = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)
        cv.Set(mask, cv.ScalarAll(255))
        
        # TODO: Is this necessary? Remove border pixels from mask
        for i in range(mask.cols):
            cv.Set2D(mask, 0, i, 0)
            cv.Set2D(mask, mask.rows - 1, i, 0)
        for j in range(mask.rows):
            cv.Set2D(mask, j, 0, 0)
            cv.Set2D(mask, j, mask.cols - 1, 0)
            
        for i in range(self.n_warps):
            theta = ran.uniform(-0.8, 0.8)
            sx = cos(theta) + ran.uniform(-0.5, 0.5)
            rx = -sin(theta) + ran.uniform(-0.5, 0.5)
            sy = sin(theta) + ran.uniform(-0.5, 0.5)
            ry = cos(theta) + ran.uniform(-0.5, 0.5)
            tx = (1 - sx) * sample.width / 2 - rx * sample.height / 2
            ty = (1 - sy) * sample.height / 2 - ry * sample.width / 2
            cv.Set2D(warp, 0, 0, sx)  
            cv.Set2D(warp, 0, 1, rx)
            cv.Set2D(warp, 0, 2, tx)
            cv.Set2D(warp, 1, 0, sy)  
            cv.Set2D(warp, 1, 1, ry)
            cv.Set2D(warp, 1, 2, ty)

            # TODO: replace WarpAffine with WarpPersective
            cv.WarpAffine(grey, result, warp, fillval=0)
            cv.WarpAffine(mask, warped_mask, warp)
                        
            eig = cv.CreateImage(cv.GetSize(grey), 32, 1)
            temp = cv.CreateImage(cv.GetSize(grey), 32, 1)
            
#            features = []
#            
#            features = cv.GoodFeaturesToTrack(result, eig, temp, self.max_count,
#                    self.quality, self.good_feature_distance, mask=None, blockSize=3, useHarris=0, k=0.04)

            (keypoints, descriptors) = cv.ExtractSURF(result, warped_mask, cv.CreateMemStorage(0), (0, 500, 3, 4))
            
            for i in range(len(keypoints)):
                descriptor = descriptors[i]
                if positive:
                    self.positive_samples.append(descriptor)
                    self.classes.append(1)
                else:
                    self.negative_samples.append(descriptor)
                    self.classes.append(-1)
                keypoint = keypoints[i][0]
                
                cv.Circle(result, (int(keypoint[0]), int(keypoint[1])), self.feature_size, (255, 255, 255, 0), cv.CV_FILLED, 8, 0)
           
            #cv.ShowImage("Warped", result)
                        
            #cv.WaitKey(self.waitkey)                    
    
    def generate_positive_samples(self, roi, reset=True):
        if reset:
            self.positive_samples = list()
            # TODO delete class values

        warp = cv.CreateMat(2, 3, cv.CV_32FC1)    
        ran = random.Random()
        sample = cv.GetSubRect(self.frame, roi)
        
#        TODO: Create mask by filling inward from outer contour.
#        sz = (sample.width & -2, sample.height & -2)
#        sz_sample = cv.CreateMat(sz[1], sz[0], cv.CV_8UC3)
#        cv.Resize(sample, sz_sample)
#        pyr = cv.CreateMat(sz[1]/2, sz[0]/2, cv.CV_8UC3)
        
        #cv.ShowImage("First Detection", sample)
        
        noise = cv.CloneMat(sample)
        grey = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)

#        cv.PyrDown(sample, pyr, 7)
#        cv.PyrUp(pyr, sample, 7)
        
        cv.Zero(noise)
        cv.RandArr(cv.RNG(), noise, cv.CV_RAND_NORMAL, (0, 0, 0), (self.noise_var, self.noise_var, self.noise_var))
    
        cv.Add(sample, noise, sample)
        cv.CvtColor(sample, grey, cv.CV_RGB2GRAY)
        
        cv.EqualizeHist(grey, grey)

#        cv.FloodFill(grey, (grey.width/2, grey.height/2), 200, lo_diff=(0, 0, 0, 0), up_diff=(100, 100, 100, 100), flags=4, mask=None)
#        edges = cv.CloneMat(grey)
#        cv.Canny(grey, edges, 0, 200)
#        cv.Dilate(grey, grey, None, 1)
#        
#        contour = cv.FindContours(edges, cv.CreateMemStorage(0), mode=cv.CV_RETR_EXTERNAL)
#        
#        cv.DrawContours(edges, contour, cv.RGB(255, 255, 255), cv.RGB(255, 255, 255), 1, 5)
#        cv.ShowImage("Edges", edges)

        result = cv.CloneMat(grey)

        mask = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)
        warped_mask = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)
        cv.Set(mask, cv.ScalarAll(255))
        
        # TODO: Is this necessary? Remove border pixels from mask
        for i in range(mask.cols):
            cv.Set2D(mask, 0, i, 0)
            cv.Set2D(mask, mask.rows - 1, i, 0)
        for j in range(mask.rows):
            cv.Set2D(mask, j, 0, 0)
            cv.Set2D(mask, j, mask.cols - 1, 0)
            
        for i in range(self.n_warps):
            theta = ran.uniform(-0.8, 0.8)
            sx = cos(theta) + ran.uniform(-0.5, 0.5)
            rx = -sin(theta) + ran.uniform(-0.5, 0.5)
            sy = sin(theta) + ran.uniform(-0.5, 0.5)
            ry = cos(theta) + ran.uniform(-0.5, 0.5)
            tx = (1 - sx) * sample.width / 2 - rx * sample.height / 2
            ty = (1 - sy) * sample.height / 2 - ry * sample.width / 2
            cv.Set2D(warp, 0, 0, sx)  
            cv.Set2D(warp, 0, 1, rx)
            cv.Set2D(warp, 0, 2, tx)
            cv.Set2D(warp, 1, 0, sy)  
            cv.Set2D(warp, 1, 1, ry)
            cv.Set2D(warp, 1, 2, ty)

            # TODO: replace WarpAffine with WarpPersective
            cv.WarpAffine(grey, result, warp, fillval=0)
            cv.WarpAffine(mask, warped_mask, warp)
                        
            eig = cv.CreateImage(cv.GetSize(grey), 32, 1)
            temp = cv.CreateImage(cv.GetSize(grey), 32, 1)
            
#            features = []
#            
#            features = cv.GoodFeaturesToTrack(result, eig, temp, self.max_count,
#                    self.quality, self.good_feature_distance, mask=None, blockSize=3, useHarris=0, k=0.04)

            (keypoints, descriptors) = cv.ExtractSURF(result, warped_mask, cv.CreateMemStorage(0), (0, 500, 3, 4))
            
            for i in range(len(keypoints)):
                descriptor = descriptors[i]
                self.positive_samples.append(descriptor)
                self.classes.append(1)
                keypoint = keypoints[i][0]
                
                cv.Circle(result, (int(keypoint[0]), int(keypoint[1])), self.feature_size, (255, 255, 255, 0), cv.CV_FILLED, 8, 0)
           
            #cv.ShowImage("Warped", result)
                        
            #cv.WaitKey(self.waitkey)
   
    def generate_negative_samples(self, sample_roi, reset=True):
        if reset:
            self.negative_samples = list()
            
        negative_rois = []
        
        x_detect, y_detect, w, h = sample_roi
        
        x_step = int(self.shift * w)
        y_step = int(self.shift * h)

        x_upper_max = self.frame.width - w - 1
        y_upper_max = self.frame.height - h - 1
        
        for x in range(0, x_upper_max, x_step):
            for y in range(0, y_upper_max, y_step):
                roi = (x, y, w, h)
                if self.box_overlap(sample_roi, roi) > self.max_overlap:
                    continue
                negative_rois.append(roi)
                
        for roi in negative_rois:
              self.generate_samples(roi, positive=False, reset=False)
              
    def box_overlap(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        if x1 > x2 + w2:
            return 0.0
        if y1 > y2 + h2:
            return 0.0            
        if x1 + w1 < x2:
            return 0.0
        if y1 + h1 < y2: 
            return 0.0

        colInt =  min(x1 + w1, x2 + w2) - max(x1, x2)
        rowInt =  min(y1 + h1, y2 + h2) - max(y1, y2)

        intersection = colInt * rowInt
        area1 = w1 * h1
        area2 = w2 * h2
        
        return float(intersection) / (area1 + area2 - intersection)
        
#    def generate_positive_examples(self, roi, image, reset):
#        if reset:
#            self.pos_ex_init.clear()
#        
#        cv.SetImageROI(image, roi)    
#        for i in range(n_warps):
#
#            warped__image = cv.WARP(image)
#            features = get_features(warped_image)
#            self.pos_ex_init.append(features)
#        cv.ResetImageROI(cv_image)
#            
#    def generate_negative_examples(self, roi, image, reset):
#        if reset:
#            self.neg_ex_init.clear()
#            
#        for i in n_warps:
#            cv.SetImageROI(image, roi)
#            warped__image = cv.WARP(image)
#            features = get_features(warped_image)
#            self.pos_ex_init.append(features)
    
    def load_detector(self, detector):
        if detector == "face":
            try:
                """ Set up the Haar face detection parameters """
                self.cascade_frontal_alt = rospy.get_param("~cascade_frontal_alt", "")
                self.cascade_frontal_alt2 = rospy.get_param("~cascade_frontal_alt2", "")
                self.cascade_profile = rospy.get_param("~cascade_profile", "")
                
                self.cascade_frontal_alt = cv.Load(self.cascade_frontal_alt)
                self.cascade_frontal_alt2 = cv.Load(self.cascade_frontal_alt2)
                self.cascade_profile = cv.Load(self.cascade_profile)
        
                self.min_size = (20, 20)
                self.image_scale = 2
                self.haar_scale = 1.5
                self.min_neighbors = 1
                self.haar_flags = cv.CV_HAAR_DO_CANNY_PRUNING
                #self.HaarMemStorage = cv.CreateMemStorage(0)
                
                return True
            except:
                return False
        else:
            return False
    
    def save_detector(self, detector):
        pass
    
    def detect_roi(self, detector):
        if detector == "face":
            detect_box = self.detect_face()
        
        return detect_box
    
    def detect_face(self):
        if not self.small_image:
            self.small_image = cv.CreateImage((cv.Round(self.frame_size[0] / self.image_scale),
                       cv.Round(self.frame_size[1] / self.image_scale)), 8, 1)
    
        """ Convert color input image to grayscale """
        cv.CvtColor(self.frame, self.grey, cv.CV_BGR2GRAY)
        
        """ Equalize the histogram to reduce lighting effects. """
        cv.EqualizeHist(self.grey, self.grey)
    
        """ Scale input image for faster processing """
        cv.Resize(self.grey, self.small_image, cv.CV_INTER_LINEAR)
    
        """ First check one of the frontal templates """
        if self.cascade_frontal_alt:
            faces = cv.HaarDetectObjects(self.small_image, self.cascade_frontal_alt, cv.CreateMemStorage(0),
                                          self.haar_scale, self.min_neighbors, self.haar_flags, self.min_size)
                                         
        """ If that fails, check the profile template """
        if not faces:
            if self.cascade_profile:
                faces = cv.HaarDetectObjects(self.small_image, self.cascade_profile, cv.CreateMemStorage(0),
                                             self.haar_scale, self.min_neighbors, self.haar_flags, self.min_size)

            if not faces:
                """ If that fails, check a different frontal profile """
                if self.cascade_frontal_alt2:
                    faces = cv.HaarDetectObjects(self.small_image, self.cascade_frontal_alt2, cv.CreateMemStorage(0),
                                         self.haar_scale, self.min_neighbors, self.haar_flags, self.min_size)
            
        if not faces:
            if self.show_text:
                hscale = 0.4 * self.frame_size[0] / 160. + 0.1
                vscale = 0.4 * self.frame_size[1] / 120. + 0.1
                text_font = cv.InitFont(cv.CV_FONT_VECTOR0, hscale, vscale, 0, 1, 8)
                cv.PutText(self.marker_image, "LOST FACE!", (50, int(self.frame_size[1] * 0.9)), text_font, cv.RGB(255, 255, 0))
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
        
    def get_features(self, track_box):         
        """ Zero the mask with all black pixels """
        cv.Zero(self.mask)

        """ Get the coordinates and dimensions of the track box """
        try:
            x,y,w,h = track_box
        except:
            return None
        
        if self.auto_face_tracking:
            """ For faces, the detect box tends to extend beyond the actual object so shrink it slightly """
            x = int(0.97 * x)
            y = int(0.97 * y)
            w = int(1 * w)
            h = int(1 * h)
            
            """ Get the center of the track box (type CvRect) so we can create the
                equivalent CvBox2D (rotated rectangle) required by EllipseBox below. """
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            roi_box = ((center_x, center_y), (w, h), 0)
            
            """ Create a filled white ellipse within the track_box to define the ROI. """
            cv.EllipseBox(self.mask, roi_box, cv.CV_RGB(255,255, 255), cv.CV_FILLED)
            
        else:
            """ For manually selected regions, just use a rectangle """
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv.Rectangle(self.mask, pt1, pt2, cv.CV_RGB(255,255, 255), cv.CV_FILLED)
        
        """ Create the temporary scratchpad images """
        eig = cv.CreateImage (cv.GetSize(self.grey), 32, 1)
        temp = cv.CreateImage (cv.GetSize(self.grey), 32, 1)

        if self.feature_type == 0:
            """ Find keypoints to track using Good Features to Track """
            self.features = cv.GoodFeaturesToTrack(self.grey, eig, temp, self.max_count,
                self.quality, self.good_feature_distance, mask=self.mask, blockSize=3, useHarris=0, k=0.04)
        
        elif self.feature_type == 1:
            """ Find keypoints to track using SURF """              
            (keypoints, descriptors) = cv.ExtractSURF(self.grey, self.mask, cv.CreateMemStorage(0), (0, self.surf_hessian_quality, 3, 1))
            
            """ Set the initial classifier keypoints and descriptors """
            self.classifier_descriptors = descriptors
            self.classifier_keypoints = keypoints
            self.descriptors = descriptors
            self.keypoints = keypoints
            
            if self.feature_type == 1:
                for keypoint in keypoints:
                    self.features.append(keypoint[0])
        
        if self.auto_min_features:
            """ Since the detect box is larger than the actual face or desired patch, shrink the number a features by 10% """
            self.min_features = int(len(self.features) * 0.9)
            self.abs_min_features = int(0.5 * self.min_features)
            
    def track_roi(self):
        feature_box = None
        
        """ Initialize intermediate images if necessary """
        if not self.pyramid:
            self.grey = cv.CreateImage(cv.GetSize(self.frame), 8, 1)
            self.prev_grey = cv.CreateImage(cv.GetSize(self.frame), 8, 1)
            self.pyramid = cv.CreateImage(cv.GetSize(self.frame), 8, 1)
            self.prev_pyramid = cv.CreateImage(cv.GetSize(self.frame), 8, 1)
            self.features = []
            
        """ Create a grey version of the image """
        cv.CvtColor(self.frame, self.grey, cv.CV_BGR2GRAY)
        
        """ Equalize the histogram to reduce lighting effects """
        cv.EqualizeHist(self.grey, self.grey)
            
        if self.track_box and self.features != []:
            """ We have feature points, so track and display them """

            """ Calculate the optical flow """
            self.features, status, track_error = cv.CalcOpticalFlowPyrLK(
                self.prev_grey, self.grey, self.prev_pyramid, self.pyramid,
                self.features,
                (self.win_size, self.win_size), 3,
                (cv.CV_TERMCRIT_ITER|cv.CV_TERMCRIT_EPS, 20, 0.01),
                self.flags)

            """ Keep only high status points """
            self.features = [ p for (st,p) in zip(status, self.features) if st]        
                                    
        
        """ Swapping the images """
        self.prev_grey, self.grey = self.grey, self.prev_grey
        self.prev_pyramid, self.pyramid = self.pyramid, self.prev_pyramid
        
        """ If we have some features... """
        if len(self.features) > 0:
            """ The FitEllipse2 function below requires us to convert the feature array
                into a CvMat matrix """
            try:
                self.feature_matrix = cv.CreateMat(1, len(self.features), cv.CV_32SC2)
            except:
                pass
                        
            """ Draw the points as green circles and add them to the features matrix """
            i = 0
            for the_point in self.features:
                if self.show_features:
                    cv.Circle(self.marker_image, (int(the_point[0]), int(the_point[1])), self.feature_size, (0, 255, 0, 0), cv.CV_FILLED, 8, 0)
                try:
                    cv.Set2D(self.feature_matrix, 0, i, (int(the_point[0]), int(the_point[1])))
                except:
                    pass
                i = i + 1
    
            """ Draw the best fit ellipse around the feature points """
            if len(self.features) > 6:
                feature_box = cv.FitEllipse2(self.feature_matrix)
            else:
                feature_box = None
                
            """ Do we also want the SURF keypoints and descriptors? """
            if feature_box and self.feature_type == 1 or ((self.feature_type == 0 and (self.get_surf_also or self.test_classifier))):            
                """ Begin with all black pixels """
                cv.Zero(self.mask)
                
                cv.EllipseBox(self.mask, feature_box, cv.CV_RGB(255,255, 255), cv.CV_FILLED)
                                
                """ Find SURF keypoints and descriptors to be used in building an online classifier """
                if self.frame_count % 20 == 0:
                    self.frame_count = 0
                    (self.keypoints, self.descriptors) = cv.ExtractSURF(self.grey, self.mask, cv.CreateMemStorage(0), (0, self.surf_hessian_quality, 3, 1))
                
                if self.show_features and self.feature_type == 0:
                    for keypoint in self.keypoints:
                        cv.Circle(self.marker_image, (int(keypoint[0][0]), int(keypoint[0][1])), self.feature_size, (255, 0, 255, 0), cv.CV_FILLED, 8, 0)
            
            """ Are we classifying the patch? """
            if self.test_classifier and (self.feature_type == 1 or self.get_surf_also):
                self.classifer_score = self.classify_patch()
            
            """ Publish the ROI for the tracked object """
            try:
                (roi_center, roi_size, roi_angle) = feature_box
            except:
                rospy.loginfo("Patch box has shrunk to zero...")
                feature_box = None
    
            if feature_box and not self.drag_start and self.is_rect_nonzero(self.track_box):
                self.ROI = RegionOfInterest()
                self.ROI.x_offset = min(self.frame_size[0], max(0, int(roi_center[0] - roi_size[0] / 2)))
                self.ROI.y_offset = min(self.frame_size[1], max(0, int(roi_center[1] - roi_size[1] / 2)))
                self.ROI.width = min(self.frame_size[0], int(roi_size[0]))
                self.ROI.height = min(self.frame_size[1], int(roi_size[1]))
                
            self.pubROI.publish(self.ROI)
            
        if feature_box is not None and len(self.features) > 0:
            return feature_box
        else:
            return None
        
    def add_features(self):
        """ Look for any new features around the current feature cloud """
        
        """ Create the ROI mask"""
        roi = cv.CreateImage(cv.GetSize(self.frame), 8, 1) 
        
        """ Begin with all black pixels """
        cv.Zero(roi)
        
        """ Get the coordinates and dimensions of the current track box """
        try:
            ((x,y), (w,h), a) = self.track_box
        except:
            rospy.loginfo("Track box has shrunk to zero...")
            return
        
        """ Expand the track box to look for new features """
        w = int(self.expand_roi * w)
        h = int(self.expand_roi * h)
        
        roi_box = ((x,y), (w,h), a)
        
        """ Create a filled white ellipse within the track_box to define the ROI. """
        cv.EllipseBox(roi, roi_box, cv.CV_RGB(255,255, 255), cv.CV_FILLED)
        
        """ Create the temporary scratchpad images """
        eig = cv.CreateImage (cv.GetSize(self.grey), 32, 1)
        temp = cv.CreateImage (cv.GetSize(self.grey), 32, 1)
        
        if self.feature_type == 0:
            """ Get the new features using Good Features to Track """
            features = cv.GoodFeaturesToTrack(self.grey, eig, temp, self.max_count,
            self.quality, self.good_feature_distance, mask=roi, blockSize=3, useHarris=0, k=0.04)
        
        elif self.feature_type == 1:
            """ Get the new features using SURF """
            features = []
            (surf_features, descriptors) = cv.ExtractSURF(self.grey, roi, cv.CreateMemStorage(0), (0, self.surf_hessian_quality, 3, 1))
            for feature in surf_features:
                features.append(feature[0])
                
        """ Append new features to the current list if they are not too far from the current cluster """
        for new_feature in features:
            try:
                distance = self.distance_to_cluster(new_feature, self.features)
                if distance > self.add_feature_distance:
                    self.features.append(new_feature)
            except:
                pass
                
        """ Remove duplicate features """
        self.features = list(set(self.features))

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
    
    def prune_features(self, min_features, outlier_threshold, mse_threshold):
        sum_x = 0
        sum_y = 0
        sum_z = 0
        sse = 0
        features_xy = self.features
        features_z = self.features
        n_xy = len(self.features)
        n_z = n_xy
        
        if self.use_depth_for_tracking:
            if not self.depth_image:
                return ((0, 0, 0), 0, 0, -1)
            else:
                (cols, rows) = cv.GetSize(self.depth_image)
        
        """ If there are no features left to track, start over """
        if n_xy == 0:
            return ((0, 0, 0), 0, 0, -1)
        
        """ Compute the COG (center of gravity) of the cluster """
        for point in self.features:
            sum_x = sum_x + point[0]
            sum_y = sum_y + point[1]
        
        mean_x = sum_x / n_xy
        mean_y = sum_y / n_xy
        
        if self.use_depth_for_tracking:
            for point in self.features:
                try:
                    z = cv.Get2D(self.depth_image, min(rows - 1, int(point[1])), min(cols - 1, int(point[0])))
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
        for point in self.features:
            sse = sse + (point[0] - mean_x) * (point[0] - mean_x) + (point[1] - mean_y) * (point[1] - mean_y)
            #sse = sse + abs((point[0] - mean_x)) + abs((point[1] - mean_y))
        
        """ Get the average over the number of feature points """
        mse_xy = sse / n_xy
        
        """ The MSE must be > 0 for any sensible feature cluster """
        if mse_xy == 0 or mse_xy > mse_threshold:
            return ((0, 0, 0), 0, 0, -1)
        
        """ Throw away the outliers based on the x-y variance """
        max_err = 0
        for point in self.features:
            std_err = ((point[0] - mean_x) * (point[0] - mean_x) + (point[1] - mean_y) * (point[1] - mean_y)) / mse_xy
            if std_err > max_err:
                max_err = std_err
            if std_err > outlier_threshold:
                features_xy.remove(point)
                try:
                	features_z.remove(point)
                	n_z = n_z - 1
                except:
                	pass
                
                n_xy = n_xy - 1
                                
        """ Now do the same for depth """
        if self.use_depth_for_tracking:
            sse = 0
            for point in features_z:
                try:
    				z = cv.Get2D(self.depth_image, min(rows - 1, int(point[1])), min(cols - 1, int(point[0])))
    				z = z[0]
    				sse = sse + (z - mean_z) * (z - mean_z)
                except:
                    n_z = n_z - 1
            
            mse_z = sse / n_z
            
            """ Throw away the outliers based on depth using percent error rather than standard error since depth
                 values can jump dramatically at object boundaries  """
            for point in features_z:
                try:
                    z = cv.Get2D(self.depth_image, min(rows - 1, int(point[1])), min(cols - 1, int(point[0])))
                    z = z[0]
                except:
                    continue
                try:
                    pct_err = abs(z - mean_z) / mean_z
                    if pct_err > self.pct_err_z:
                        features_xy.remove(point)
                except:
                    pass
        else:
            mse_z = -1
        
        self.features = features_xy
               
        """ Consider a cluster bad if we have fewer than abs_min_features left """
        if len(self.features) < self.abs_min_features:
            score = -1
        else:
            score = 1

        return ((mean_x, mean_y, mean_z), mse_xy, mse_z, score)

    def get_template(self, roi): 
        if len(roi) == 3:
            (center, size, angle) = roi
            pt1 = (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2))
            pt2 = (int(center[0] + size[0] / 2), int(center[1] + size[1] / 2))
            w = pt2[0] - pt1[0]
            h = pt2[1] - pt1[1]
            roi = (pt1[0], pt1[1], w, h)
        
        try:    
            template_header = cv.GetSubRect(self.frame, roi)
            template = cv.CreateImage((template_header.cols, template_header.rows), cv.IPL_DEPTH_8U, 3)
            cv.Copy(template_header, template)
    
            cv.NamedWindow("Template", cv.CV_NORMAL)
            cv.ResizeWindow("Template", 320, 240)
            cv.MoveWindow("Template", 700, 50)
            cv.ShowImage("Template", template)
            
            return (template, roi)
        except:
            rospy.loginfo("Exception getting template!")
            return (None, None)
    
    def get_template2(self, roi): 
        if len(roi) == 3:
            (center, size, angle) = roi
            pt1 = (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2))
            pt2 = (int(center[0] + size[0] / 2), int(center[1] + size[1] / 2))
            roi = (pt1[0], pt1[1], pt2[0] - pt1[0], pt2[1] - pt1[1])
            
        raw_template = cv.GetSubRect(self.frame, roi)
        
        self.n_samples += 1
        if self.n_samples == 1:
            template = cv.CreateMat(roi[3], roi[2], cv.CV_8UC3)
            tmp_sum = cv.CreateMat(roi[3], roi[2], cv.CV_8UC3)
            cv.Resize(raw_template, template)    
            self.template = template
            self.template_sum = template
        else:
            template = cv.CreateMat(self.template.rows, self.template.cols, cv.CV_8UC3)
            tmp_sum = cv.CreateMat(self.template.rows, self.template.cols, cv.CV_8UC3)
            cv.Resize(raw_template, template)
            self.template = template
            cv.ConvertScale(self.template_sum, tmp_sum, float(self.n_samples) / (self.n_samples + 1))
            cv.ScaleAdd(template, 1.0 / (self.n_samples + 1), tmp_sum, self.template_sum)
        cv.ShowImage("Template", self.template_sum)
        cv.ResizeWindow("Template", roi[2] * 2, roi[3] * 2)
        
        return (roi, raw_template)
                
    def get_descriptors(self, roi):
        cv.CvtColor(self.frame, self.grey, cv.CV_BGR2GRAY)
        cv.EqualizeHist(self.grey, self.grey)
        mask = cv.CreateImage(cv.GetSize(self.frame), 8, 1) 
        cv.Zero(mask)
        
        if len(roi) == 3:
            (center, size, angle) = roi
            pt1 = (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2))
            pt2 = (int(center[0] + size[0] / 2), int(center[1] + size[1] / 2))
        elif len(roi) == 4:    
            x,y,w,h = roi
            pt1 = (x, y)
            pt2 = (x + w, y + h)
        else:
            return
        
        cv.Rectangle(mask, pt1, pt2, cv.CV_RGB(255,255, 255), cv.CV_FILLED)
        
        self.n_samples += 1
            
        """ Find SURF keypoints and descriptors in the current roi """              
        (keypoints, descriptors) = cv.ExtractSURF(self.grey, mask, cv.CreateMemStorage(0), (0, self.surf_hessian_quality, 3, 1))
        
        if self.show_features:
            x,y,h,w = roi
            for keypoint in keypoints:
                keypoint_x = int(keypoint[0][0]) - x
                keypoint_y = int(keypoint[0][1]) - y
                cv.Circle(self.template, (keypoint_x, keypoint_y), self.feature_size, (255, 0, 255, 0), cv.CV_FILLED, 8, 0)
                cv.ShowImage("Template", self.template)
                cv.Circle(self.marker_image, (int(keypoint[0][0]), int(keypoint[0][1])), self.feature_size, (255, 0, 255, 0), cv.CV_FILLED, 8, 0)                   

        return [keypoints, descriptors]

    def match_template(self, image):
        image_ipl = cv.CreateImage((image.cols, image.rows), cv.IPL_DEPTH_8U, 3)
        template_ipl = cv.CreateImage((self.template_sum.cols, self.template_sum.rows), cv.IPL_DEPTH_8U, 3)
        cv.Copy(image, image_ipl)
        cv.Copy(self.template_sum, template_ipl)
        
        try:
            if not self.match_box:
                self.match_box = self.track_box
            if self.is_rect_nonzero(self.match_box):
                (center, size, angle) = self.match_box
                roi_w = min(template_ipl.width * 2, int(size[0] * 4))
                roi_h = min(template_ipl.height * 2, int(size[1] * 4))
                roi_x = max(0, int(center[0] - roi_w / 2))
                roi_y = max(0, int(center[1] - roi_h / 2))
                roi_x2 = min(image_ipl.width, roi_x + roi_w)
                roi_y2 = min(image_ipl.height, roi_y + roi_h)
                roi_w = roi_x2 - roi_x
                roi_h = roi_y2 - roi_y
            else:
                rospy.loginfo("Scanning whole image...")
                roi_x = 0
                roi_y = 0
                roi_w = image_ipl.width
                roi_h = image_ipl.height
        except:
            roi_x = 0
            roi_y = 0
            roi_w = image_ipl.width
            roi_h = image_ipl.height
            
        roi = (roi_x, roi_y, roi_w, roi_h)
        cv.SetImageROI(image_ipl, roi)

        W,H = cv.GetSize(image_ipl)
        w,h = cv.GetSize(template_ipl)
        width = W - w + 1
        height = H - h + 1
        try:
            result = cv.CreateImage((width, height), 32, 1)
            cv.MatchTemplate(image_ipl, template_ipl, result, cv.CV_TM_CCOEFF_NORMED)
            (min_score, max_score, minloc, maxloc) = cv.MinMaxLoc(result)
            if max_score < 0.7:
                return None
            (x, y) = maxloc
            corrected_x = int(x) + roi_x
            corrected_y = int(y) + roi_y
    
            match_box = (corrected_x, corrected_y, w, h)
            cv.Rectangle(self.marker_image, (corrected_x, corrected_y), (corrected_x + w, corrected_y + h),(255, 255, 0), 3, 0)
            cv.ResetImageROI(image_ipl)
            return match_box
        except:
            return roi
    
    def evaluate_roi(self, roi):
        return self.evaluate_templates(roi)
    
    def evaluate_templates(self, roi):
        if len(roi) == 4:
            roi = self.cvRect_to_cvBox2D(roi)

        (center, size, angle) = roi
        # Expand the ROI so that the template will fit.
        size = (int(2 * size[0]), int(2 * size[1]))
        pt1 = (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2))
        pt2 = (int(center[0] + size[0] / 2), int(center[1] + size[1] / 2))
        w = pt2[0] - pt1[0]
        h = pt2[1] - pt1[1]
        roi = (pt1[0], pt1[1], w, h)
        
        try:    
            patch_image = cv.GetSubRect(self.frame, roi)
        except:
            return False
        
        patch_image_ipl = cv.CreateImage((patch_image.cols, patch_image.rows), cv.IPL_DEPTH_8U, 3)
        patch_image_ipl_grey = cv.CreateImage((patch_image.cols, patch_image.rows), cv.IPL_DEPTH_8U, 1)
        
        # Cycle through the current templates.
        for template in self.templates:
            template_ipl = cv.CreateImage((template.cols, template.rows), cv.IPL_DEPTH_8U, 3)
            template_ipl_grey = cv.CreateImage((template.cols, template.rows), cv.IPL_DEPTH_8U, 1)
            cv.Copy(patch_image, patch_image_ipl)
            cv.Copy(template, template_ipl)
            cv.CvtColor(patch_image_ipl, patch_image_ipl_grey, cv.CV_RGB2GRAY)
            cv.CvtColor(template_ipl, template_ipl_grey, cv.CV_RGB2GRAY)

            W,H = cv.GetSize(patch_image_ipl)
            w,h = cv.GetSize(template_ipl)
            width = W - w + 1
            height = H - h + 1
            if width <= 0 or height <= 0:
                continue
    
            result = cv.CreateImage((width, height), 32, 1)
    
            try:
                cv.MatchTemplate(patch_image_ipl_grey, template_ipl_grey, result, cv.CV_TM_CCOEFF_NORMED)
            except:
                continue
            (min_score, max_score, minloc, maxloc) = cv.MinMaxLoc(result)
            rospy.loginfo("Max Score: " + str(max_score))
            if max_score > 0.99:
                return True
            
        # If we get no matches, return False
        return False
    
    def redetect_roi(self, image):
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
            
    def classify_patch(self):
        """ Compare descriptor sets beteween target and test """
        
        n_target = len(self.classifier_descriptors)
        n_test = len(self.descriptors)

        votes = 0
        for i in range(n_target):
            max_dp = 0
            for j in range(n_test):
                if self.classifier_keypoints[i][1] !=  self.keypoints[j][1]:
                    continue
                dp =  self.dot_product(self.classifier_descriptors[i], self.descriptors[j])
                if dp > max_dp:
                    max_dp = dp
            if max_dp > 0.9:
                votes = votes + 1
            else:
                votes = votes -1
                    
        rospy.loginfo("VOTES: " + str(votes) + " out of " + str(n_target))
        return float(votes / n_target)
        
    def dot_product(self, vec1, vec2):
        dim = len(vec1)
        dp = 0
        for i in range(dim):
            dp += vec1[i] * vec2[i]
            
        return dp
    
    def cvRect_to_cvBox2D(self, roi):
        if len(roi) == 3:
            (center, size, angle) = roi
            pt1 = (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2))
            pt2 = (int(center[0] + size[0] / 2), int(center[1] + size[1] / 2))
            rect = (pt1[0], pt1[1], pt2[0] - pt1[0], pt2[1] - pt1[1])
        else:
            (p1_x, p1_y, width, height) = roi
            center = (int(p1_x + width / 2), int(p1_y + height / 2))
            size = (width, height)
            angle = 0
            rect = (center, size, angle)
            
        return rect
        

class RTrees():
    def __init__(self):
        self.model = cv2.RTrees()

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL], np.uint8)
        #CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));
        params = dict(max_depth=10 )
        self.model.train(samples, cv2.CV_ROW_SAMPLE, responses, varType = var_types, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )
        
class KNearest():
    def __init__(self):
        self.model = cv2.KNearest()

    def train(self, samples, responses):
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, k = 10)
        return results.ravel()

class Boost():
    def __init__(self):
        self.model = cv2.Boost()
        self.class_n = 26
    
    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_samples = self.unroll_samples(samples)
        new_responses = self.unroll_responses(responses)
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL, cv2.CV_VAR_CATEGORICAL], np.uint8)
        #CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 )
        params = dict(max_depth=5) #, use_surrogates=False)
        self.model.train(new_samples, cv2.CV_ROW_SAMPLE, new_responses, varType = var_types, params=params)

    def predict(self, samples):
        new_samples = self.unroll_samples(samples)
        pred = np.array( [self.model.predict(s, returnSum = True) for s in new_samples] )
        pred = pred.reshape(-1, self.class_n).argmax(1)
        return pred

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples
    
    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*self.class_n, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        new_responses[resp_idx] = 1
        return new_responses

class SVM():
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        params = dict( kernel_type = cv2.SVM_LINEAR, 
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )

def main(args):
    """ Display a help message if appropriate """
    help_message =  "Hot keys: \n" \
          "\tq - quit the program\n" \
          "\tc - delete current features\n" \
          "\tt - toggle text captions on/off\n" \
          "\tf - toggle display of features on/off\n" \
          "\tn - toggle \"night\" mode on/off\n" \
          "\ta - toggle auto face tracking on/off\n" \
          "\tm - match current face template\n"

    print help_message
    
    """ Fire up the Video Tracker node """
    VT = VideoTracker("pi_video_tracker")

    try:
      rospy.spin()
    except KeyboardInterrupt:
      print "Shutting down video tracker node."
      cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)