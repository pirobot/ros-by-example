#!/usr/bin/env python
# coding=UTF-8

""" camshift_node.py - Version 1.0 2011-04-19

    Modification of the ROS OpenCV Camshift example using cv_bridge and publishing the ROI
    coordinates to the /roi topic.   
"""
# rosrun video_stream_opencv video_stream _fps:=30 _camera_name:=videofile _video_stream_provider:=/mnt/data/workspace/ros/src/blob-tracking/blob-tracking/src/media/2018-06-14-16.35.22.mp4
# ./camshift.py input_rgb_image:=/camera

import roslib; roslib.load_manifest('rbx1_vision')
import rospy
import numpy as np
import cv2 as cv2
from ros2opencv2 import ROS2OpenCV2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from blob_tracking.msg import HSV_history
from blob_tracking.msg import HSV

class CamShiftNode(ROS2OpenCV2):
    def __init__(self, node_name):

        self.node_name = node_name
        
        # The minimum saturation of the tracked color in HSV space,
        # as well as the min and max value (the V in HSV) and a 
        # threshold on the backprojection probability image.
        self.smin = rospy.get_param("~smin", 85)
        self.vmin = rospy.get_param("~vmin", 50)
        self.vmax = rospy.get_param("~vmax", 254)
        self.threshold = rospy.get_param("~threshold", 50)
                       
        #self.bin_count = 16
        #self.bin_w     = 24
        self.bin_count = 180
        self.bin_w     = 2

        # Initialize a number of variables
        self.hist = None
        self.track_window = None
        self.show_backproj = False
        ROS2OpenCV2.__init__(self, node_name)
	self.hsv_history_sub = rospy.Subscriber("adaptive_hsv_thresholds", HSV_history, self.hsv_history_callback)
	print("Init done...")
    
    # These are the callbacks for the slider controls
    def set_smin(self, pos):
	self.smin = pos
	print("set_smin()", self.smin, self.smax, self.vmin, self.vmax)
        
    def set_vmin(self, pos):
	self.vmin = pos
	print("set_vmin()", self.smin, self.smax, self.vmin, self.vmax)
            
    def set_vmax(self, pos):
	self.vmax = pos
	print("set_vmax()", self.smin, self.smax, self.vmin, self.vmax)
       
    def set_threshold(self, pos):
        self.threshold = pos


    def hsv_history_callback(self, hsv_history):
	history_len = len(hsv_history.history)
	#print(history_len)
	#print(hsv_history.history[history_len-1])
	self.last_hsv = hsv_history.history[history_len-1]
	self.smin = self.last_hsv.sat.left
	self.smax = self.last_hsv.sat.right
	self.vmin = self.last_hsv.val.left
	self.vmax = self.last_hsv.val.right
	#self.hist
	tr = 1.0 * self.bin_count / 180.0
	hue_min = self.last_hsv.val.left  * tr
	hue_med = self.last_hsv.val.ch    * tr
	hue_max = self.last_hsv.val.right * tr
	mu, sigma = hue_med, 1.0*(hue_med-hue_min)/4 # mean and standard deviation
	print(mu, sigma)
	s = np.random.normal(mu, sigma, 10000)
	self.hist = np.histogram(s, bins=180, range=(1,180), normed=True)[0]
	#plt.plot(self.hist)
	
		
	print("hsv_history_callback()", self.smin, self.smax, self.vmin, self.vmax)
	'''
                #> rosmsg show blob_tracking/HSV_history 
                std_msgs/Header header
                  uint32 seq
                  time stamp
                  string frame_id
                blob_tracking/HSV[] history
                  blob_tracking/HSVth hue
                    uint8 ch
                    uint8 left
                    uint8 right
                    uint8 width
                  blob_tracking/HSVth sat
                    uint8 ch
                    uint8 left
                    uint8 right
                    uint8 width
                  blob_tracking/HSVth val
                    uint8 ch
                    uint8 left
                    uint8 right
                    uint8 width
	'''


    # The main processing function computes the histogram and backprojection
    def process_image(self, cv_image):
	#print("Processing...")

	# Moving these namedWindow() calls in the callback because of this bug:
	# http://answers.opencv.org/question/160607/imshow-call-never-returns-opencv-320-dev/

        # Create a number of windows for displaying the histogram,
        # parameters controls, and backprojection image
        cv2.namedWindow("Histogram", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Histogram", 700, 50)
        cv2.namedWindow("Parameters", 0)
        cv2.moveWindow("Parameters", 700, 325)
        cv2.namedWindow("Backproject", 0)
        cv2.moveWindow("Backproject", 700, 600)
        cv2.namedWindow("Mask", 0)
        cv2.moveWindow("Mask", 900, 500)
	cv2.resizeWindow("Mask", 640, 480)
        
        # Create the slider controls for saturation, value and threshold
        cv2.createTrackbar("Saturation", "Parameters", self.smin, 255, self.set_smin)
        cv2.createTrackbar("Min Value", "Parameters", self.vmin, 255, self.set_vmin)
        cv2.createTrackbar("Max Value", "Parameters", self.vmax, 255, self.set_vmax)
        cv2.createTrackbar("Threshold", "Parameters", self.threshold, 255, self.set_threshold)
        
        # First blue the image
        frame = cv2.blur(cv_image, (5, 5))
        
        # Convert from RGB to HSV spave
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask using the current saturation and value parameters
	print("cv2.inRange()", self.smin, self.smax, self.vmin, self.vmax)
        mask = cv2.inRange(hsv, np.array((0., self.smin, self.vmin)), np.array((180., 255., self.vmax)))
	#print("mask:", mask)
        cv2.imshow("Mask", mask)
        
        # If the user is making a selection with the mouse, 
        # calculate a new histogram to track
        if self.selection is not None:
            x0, y0, w, h = self.selection
            x1 = x0 + w
            y1 = y0 + h
            self.track_window = (x0, y0, x1, y1)
            hsv_roi = hsv[y0:y1, x0:x1]
            mask_roi = mask[y0:y1, x0:x1]
            #self.hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
            self.hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [self.bin_count], [0, 180] )
            #print(self.hist)
            cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX);
            #print(self.hist)
            self.hist = self.hist.reshape(-1)
            self.show_hist()

        if self.detect_box is not None:
            self.selection = None
        
        # If we have a histogram, tracking it with CamShift
        if self.hist is not None:
            # Compute the backprojection from the histogram
            backproject = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
            
            # Mask the backprojection with the mask created earlier
            backproject &= mask

            # Threshold the backprojection
            ret, backproject = cv2.threshold(backproject, self.threshold, 255, cv2.THRESH_TOZERO)

            if self.track_window is not None:
            	x, y, w, h = self.track_window
            if self.track_window is None or w <= 0 or h <=0:
                self.track_window = 0, 0, self.frame_width - 1, self.frame_height - 1
            
            # Set the criteria for the CamShift algorithm
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            
            # Run the CamShift algorithm
            self.track_box, self.track_window = cv2.CamShift(backproject, self.track_window, term_crit)

            # Display the resulting backprojection
            cv2.imshow("Backproject", backproject)

        return cv_image
        
    def show_hist(self):
        #bin_count = self.hist.shape[0]
        #bin_w = 24
        img = np.zeros((256, self.bin_count*self.bin_w, 3), np.uint8)
        for i in xrange(self.bin_count):
            h = int(self.hist[i])
            top, left     = (i*self.bin_w+2, 255)
            bottom, right = ((i+1)*self.bin_w-2, 255-h)
            hue           = int(180.0*i/self.bin_count)
            #cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
            cv2.rectangle(img, (top, left), (bottom, right), (int(180.0*i/self.bin_count), 255, 255), -1)
            if h > 20:
		print("hue: ", hue, "- h:", h, "-", top, left, bottom, right)
		print(self.bin_count)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('Histogram', img)
        

    def hue_histogram_as_image(self, hist):
            """ Returns a nice representation of a hue histogram """
            histimg_hsv = cv.CreateImage((320, 200), 8, 3)

            mybins = cv.CloneMatND(hist.bins)
            cv.Log(mybins, mybins)
            (_, hi, _, _) = cv.MinMaxLoc(mybins)
            cv.ConvertScale(mybins, mybins, 255. / hi)
    
            w,h = cv.GetSize(histimg_hsv)
            hdims = cv.GetDims(mybins)[0]
            for x in range(w):
                xh = (180 * x) / (w - 1)  # hue sweeps from 0-180 across the image
                val = int(mybins[int(hdims * x / w)] * h / 255)
                cv2.rectangle(histimg_hsv, (x, 0), (x, h-val), (xh,255,64), -1)
                cv2.rectangle(histimg_hsv, (x, h-val), (x, h), (xh,255,255), -1)
    
            histimg = cv2.cvtColor(histimg_hsv, cv.CV_HSV2BGR)
            
            return histimg
         

if __name__ == '__main__':
    try:
        node_name = "camshift"
        CamShiftNode(node_name)
        try:
            rospy.init_node(node_name)
        except:
            pass
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()

