#!/usr/bin/env python

""" camshift.py - Version 1.0 2012-02-11

    Based on the OpenCV CamShift sample code 
"""

import roslib
roslib.load_manifest('pi_video_tracker')
import rospy
from ros2opencv2 import ROS2OpenCV2
import sys
import cv
from std_msgs.msg import String
from sensor_msgs.msg import RegionOfInterest, CameraInfo

class CamShiftNode(ROS2OpenCV2):
    def __init__(self, node_name):
        ROS2OpenCV2.__init__(self, node_name)
        
        self.node_name = node_name
        
        """ Set up a smaller window to display the CamShift histogram. """
        cv.NamedWindow("Histogram", 0)
        cv.MoveWindow("Histogram", 700, 10)

        self.detect_box = None
        self.track_box = None

        self.hist = cv.CreateHist([180], cv.CV_HIST_ARRAY, [(0,180)], 1 )
        self.backproject_mode = False
        self.backproject = None
        self.hsv = None
        self.hue = None

    def process_image(self, cv_image):
        """ Convert to HSV and keep the hue """
        if not self.hsv:
           self.hsv = cv.CreateImage(self.frame_size, 8, 3)
           
        cv.CvtColor(cv_image, self.hsv, cv.CV_BGR2HSV)
        
        if not self.hue:
           self.hue = cv.CreateImage(self.frame_size, 8, 1)
           
        cv.Split(self.hsv, self.hue, None, None, None)
        
        """ Create a greyscale image for the back projection """
        if not self.backproject:
           self.backproject = cv.CreateImage(self.frame_size, 8, 1)
           
        """ If mouse is pressed, highlight the current selected rectangle
            and recompute the histogram """
        if self.drag_start and self.is_rect_nonzero(self.selection):
            sub = cv.GetSubRect(cv_image, self.selection)
            save = cv.CloneMat(sub)
            cv.ConvertScale(cv_image, cv_image, 0.5)
            cv.Copy(save, sub)

            sel = cv.GetSubRect(self.hue, self.selection )
            cv.CalcArrHist( [sel], self.hist, 0)
            (_, max_val, _, _) = cv.GetMinMaxHistValue(self.hist)
            if max_val != 0:
                cv.ConvertScale(self.hist.bins, self.hist.bins, 255. / max_val)
                
            cv.ShowImage("Histogram", self.hue_histogram_as_image(self.hist))


        if self.detect_box:
            if not self.track_box:
                self.track_box = self.detect_box
            self.track_box = self.run_camshift(cv_image)
        
#        elif self.track_box and  self.is_rect_nonzero(self.track_box):
#            self.track_box = self.detect_box
#            self.ROI = RegionOfInterest()
#            self.ROI.x_offset = min(self.frame_size[0], max(0, int(roi_center[0] - roi_size[0] / 2)))
#            self.ROI.y_offset = min(self.frame_size[1], max(0, int(roi_center[1] - roi_size[1] / 2)))
#            self.ROI.width = min(self.frame_size[0], int(roi_size[0]))
#            self.ROI.height = min(self.frame_size[1], int(roi_size[1]))
#            
#            self.pubROI.publish(self.ROI)
        
        """ Toggle between the normal and back projected image if user hits the 'b' key """
        c = cv.WaitKey(7) % 0x100
        if c == 27:
            return
        elif c == ord("b"):
            self.backproject_mode = not self.backproject_mode
        
        if not self.backproject_mode:
            return cv_image
        else:
            return backproject
        
    def run_camshift(self, cv_image):
        smin = 63
        vmin = 70
        vmax = 256
        
        #cv.InRangeS(self.hue, cv.Scalar(0, smin, min(vmin, vmax), 0), cv.Scalar(180, 256, max(vmin, vmax) ,0), self.hue)
        cv.InRangeS(self.hue, smin, vmin, self.hue)
        
        """ Run the CamShift algorithm """
        cv.CalcArrBackProject( [self.hue], self.backproject, self.hist )
        
        if self.track_box and self.is_rect_nonzero(self.track_box):
            crit = ( cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 1)
            (iters, (area, value, rect), feature_box) = cv.CamShift(self.backproject, self.track_box, crit)
            #self.track_box = rect
            #(roi_center, roi_size, roi_angle) = feature_box
            
            
        return self.cvRect_to_cvBox2D(feature_box)
        

    def hue_histogram_as_image(self, hist):
            """ Returns a nice representation of a hue histogram """
    
            histimg_hsv = cv.CreateImage((320,200), 8, 3)
            
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

def main(args):
      CS = CamShiftNode("camshift_node")
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print "Shutting down CamShift node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
