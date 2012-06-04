#!/usr/bin/env python

import cv2
from cv2 import cv as cv
import numpy as np

def is_rect_nonzero(r):
    (_,_,w,h) = r
    return (w > 0) and (h > 0)

class CamShiftDemo:
    
    def __init__(self):
        static_file = "/home/patrick/Dropbox/Robotics/ros/pi-robot-ros-pkg/experimental/ros_by_example/rbx_pi_vision/pi_video_hbrc_talk/scripts/test_images/apples_in_field.jpg"
        self.static_image = cv.LoadImage(static_file)
        #self.capture = cv.CaptureFromCAM(0)
        cv.NamedWindow( "CamShiftDemo", 1 )
        cv.NamedWindow( "Histogram", 1 )
        cv.SetMouseCallback( "CamShiftDemo", self.on_mouse)
        
        self.smin = 100 #31
        self.vmin = 100 #41
        self.vmax = 254 #255
        
        cv.NamedWindow("Parameters", 0)
        cv.CreateTrackbar("Saturation", "Parameters", self.smin, 255, self.set_smin)
        cv.CreateTrackbar("Min Value", "Parameters", self.vmin, 255, self.set_vmin)
        cv.CreateTrackbar("Max Value", "Parameters", self.vmax, 255, self.set_vmax)

        self.drag_start = None      # Set to (x,y) when mouse starts drag
        self.track_window = None    # Set to rect when the mouse drag finishes

        print( "Keys:\n"
            "    ESC - quit the program\n"
            "    b - switch to/from backprojection view\n"
            "To initialize tracking, drag across the object with the mouse\n" )
        
    def set_smin(self, pos):
        self.smin = pos
        
    def set_vmin(self, pos):
        self.vmin = pos
        
    def set_vmax(self, pos):
       self.vmax = pos

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

    def on_mouse(self, event, x, y, flags, param):
        if event == cv.CV_EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if event == cv.CV_EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = self.selection
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax - xmin, ymax - ymin)

    def run(self):
        hist = cv.CreateHist([180], cv.CV_HIST_ARRAY, [(0,180)], 1 )
        backproject_mode = False
        while True:
            #frame = cv.QueryFrame( self.capture )
            frame = cv.CreateImage(cv.GetSize(self.static_image), 8, 3)
            cv.Copy(self.static_image, frame)
            mask = cv.CreateImage(cv.GetSize(frame), 8, 1)

            # Convert to HSV and keep the hue
            hsv = cv.CreateImage(cv.GetSize(frame), 8, 3)
            cv.CvtColor(frame, hsv, cv.CV_BGR2HSV)
            
            cv.InRangeS(hsv, cv.Scalar(0, self.smin, self.vmin), cv.Scalar(180, 255, self.vmax), mask)
            
            hue = cv.CreateImage(cv.GetSize(frame), 8, 1)
            cv.Split(hsv, hue, None, None, None)
            
            hue_masked = cv.CreateImage(cv.GetSize(frame), 8, 1)
            cv.Zero(hue_masked)
            cv.Copy(hue, hue_masked, mask=mask)
            cv.ShowImage("Hue Masked", hue_masked)
            #cv.Copy(hue, hue_masked)

            # Compute back projection
            backproject_raw = cv.CreateImage(cv.GetSize(frame), 8, 1)
            backproject = cv.CreateImage(cv.GetSize(frame), 8, 1)
            cv.Zero(backproject)

            # Run the cam-shift
            cv.CalcArrBackProject( [hue_masked], backproject_raw, hist )
            cv.Copy(backproject_raw, backproject, mask=mask )


            if self.track_window and is_rect_nonzero(self.track_window):
                crit = ( cv.CV_TERMCRIT_EPS | cv.CV_TERMCRIT_ITER, 10, 1)
                (iters, (area, value, rect), track_box) = cv.CamShift(backproject, self.track_window, crit)
                self.track_window = rect
                print track_box

            # If mouse is pressed, highlight the current selected rectangle
            # and recompute the histogram

            if self.drag_start and is_rect_nonzero(self.selection):
                sub = cv.GetSubRect(frame, self.selection)
                save = cv.CloneMat(sub)
                cv.ConvertScale(frame, frame, 0.5)
                cv.Copy(save, sub)
                x,y,w,h = self.selection
                cv.Rectangle(frame, (x,y), (x+w,y+h), (255,255,255))

                sel = cv.GetSubRect(hue_masked, self.selection )
                mask_sel = cv.GetSubRect(mask, self.selection)
                cv.CalcArrHist( [sel], hist, 0, mask=mask_sel)
                (_, max_val, _, _) = cv.GetMinMaxHistValue(hist)
                if max_val != 0:
                    cv.ConvertScale(hist.bins, hist.bins, 255. / max_val)
            elif self.track_window and is_rect_nonzero(self.track_window):
                cv.EllipseBox( frame, track_box, cv.CV_RGB(255,255,255), 3, cv.CV_AA, 0 )

            if not backproject_mode:
                cv.ShowImage( "CamShiftDemo", frame )
            else:
                cv.ShowImage( "CamShiftDemo", backproject)
            cv.ShowImage( "Histogram", self.hue_histogram_as_image(hist))

            c = cv.WaitKey(100)
            if c == 27:
                break
            elif c == ord("b"):
                backproject_mode = not backproject_mode

if __name__=="__main__":
    demo = CamShiftDemo()
    demo.run()
