#!/usr/bin/env python

""" camshift_node.py - Version 1.0 2011-04-19

    Modification of the ROS OpenCV Camshift example using cv_bridge and publishing the ROI
    coordinates to the /roi topic.   
"""

import roslib
roslib.load_manifest('pi_head_tracking_tutorial')
import sys
import rospy
from cv2 import cv as cv
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class CamShiftNode:
    def __init__(self):
        rospy.init_node('pi_video_tracker')
        
        self.ROI = rospy.Publisher("roi", RegionOfInterest)

        """ Create the display window """
        cv.NamedWindow('CamShift', cv.CV_WINDOW_NORMAL)
        cv.SetMouseCallback('CamShift', self.on_mouse)
        
        cv.NamedWindow("Histogram", cv.CV_WINDOW_NORMAL)
        cv.MoveWindow("Histogram", 700, 50)
        
        """ Create the cv_bridge object """
        self.bridge = CvBridge()
        
        """ Subscribe to the raw camera image topic """
        self.image_sub = rospy.Subscriber("input", Image, self.image_callback)
        
        self.smin = 72 #31
        self.vmin = 54 #41
        self.vmax = 255 #255
        
        cv.NamedWindow("Parameters", 0)
        cv.CreateTrackbar("Saturation", "Parameters", self.smin, 255, self.set_smin)
        cv.CreateTrackbar("Min Value", "Parameters", self.vmin, 255, self.set_vmin)
        cv.CreateTrackbar("Max Value", "Parameters", self.vmax, 255, self.set_vmax)

        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.show_backproj = False
        
    def set_smin(self, pos):
        self.smin = pos
        
    def set_vmin(self, pos):
        self.vmin = pos
        
    def set_vmax(self, pos):
       self.vmax = pos
       
    def on_mouse2(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0
        if self.drag_start: 

            if flags & cv2.EVENT_FLAG_LBUTTON:
                h, w = self.frame.shape[:2]
                xo, yo = self.drag_start
                x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))
                self.selection = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.selection = (x0, y0, x1, y1)
            else:
                self.drag_start = None
                if self.selection is not None:
                    self.tracking_state = 1
                    
    def on_mouse(self, event, x, y, flags, param):
        if event == cv.CV_EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0
        if event == cv.CV_EVENT_LBUTTONUP:
            self.drag_start = None
            #self.track_window = self.selection
            self.tracking_state = 1
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)

    def image_callback(self, data):
        """ Convert the raw image to OpenCV format using the convert_image() helper function """
        cv_image = self.convert_image(data)
        
        self.frame = np.array(cv_image, dtype=np.uint8)
                
        """ Apply the CamShift algorithm using the do_camshift() helper function """        
        if self.selection:
            x0, y0, x1, y1 = self.selection
            cv2.rectangle(self.frame, (x0, y0), (x1, y1), cv.RGB(255, 255, 0), 2)
        
        #self.do_camshift()
        
        vis = self.frame.copy()
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., self.smin, self.vmin)), np.array((180., 255., self.vmax)))
        
        result = cv.CreateMat(cv_image.rows, cv_image.cols, cv_image.type)
        cv.Copy(cv_image, result, cv.fromarray(mask))
        cv.ShowImage("Result", result)
        #cv2.imshow('CamShift', vis)

        ch = cv2.waitKey(5)
        if ch == 27:
            return
        if ch == ord('b'):
            self.show_backproj = not self.show_backproj
          
    def convert_image(self, ros_image):
        try:
          cv_image = self.bridge.imgmsg_to_cv(ros_image, "bgr8")
          return cv_image
        except CvBridgeError, e:
          print e

    def do_camshift(self):
        vis = self.frame.copy()
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., self.smin, self.vmin)), np.array((180., 255., self.vmax)))
        
        if self.selection:
            x0, y0, x1, y1 = self.selection
            self.track_window = (x0, y0, x1-x0, y1-y0)
            hsv_roi = hsv[y0:y1, x0:x1]
            mask_roi = mask[y0:y1, x0:x1]
            hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
            self.hist = hist.reshape(-1)
            self.show_hist()
           
            vis_roi = vis[y0:y1, x0:x1]
            cv2.bitwise_not(vis_roi, vis_roi)
            vis[mask == 0] = 0

        if self.tracking_state == 1:
            self.selection = None
            prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
            prob &= mask
            term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
           
            if self.show_backproj:
                vis[:] = prob[...,np.newaxis]
            try: cv2.ellipse(vis, track_box, cv.RGB(255, 0, 0), 2)
            except: print track_box
            
            cv2.imshow('Tracking', vis)

        
    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('Histogram', img)
        

    def hue_histogram_as_image(self, hist):
            """ Returns a nice representation of a hue histogram """
    
            histimg_hsv = cv.CreateImage( (320,200), 8, 3)
            
            mybins = cv.CloneMatND(hist.bins)
            cv.Log(mybins, mybins)
            (_, hi, _, _) = cv.MinMaxLoc(mybins)
            cv.ConvertScale(mybins, mybins, 255. / hi)
    
            w,h = cv.GetSize(histimg_hsv)
            hdims = cv.GetDims(mybins)[0]
            for x in range(w):
                xh = (180 * x) / (w - 1)  # hue sweeps from 0-180 across the image
                val = int(mybins[int(hdims * x / w)] * h / 255)
                cv.Rectangle( histimg_hsv, (x, 0), (x, h-val), (xh,255,64), -1)
                cv.Rectangle( histimg_hsv, (x, h-val), (x, h), (xh,255,255), -1)
    
            histimg = cv.CreateImage( (320,200), 8, 3)
            cv.CvtColor(histimg_hsv, histimg, cv.CV_HSV2BGR)
            return histimg

def is_rect_nonzero(r):
    (_,_,w,h) = r
    return (w > 0) and (h > 0)           

def main(args):
      cs = CamShiftNode()
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    
    
