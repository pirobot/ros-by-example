#!/usr/bin/env python

""" ros2opencv2.py - Version 0.1 2011-04-24

    A ROS-to-OpenCV node that uses cv_bridge to map a ROS image topic and optionally a ROS
    depth image topic to the equivalent OpenCV image stream(s).
    
    Includes variables and helper functions to store detection and tracking information and display
    markers on the image.
    
    Creates an ROI publisher to publish the region of interest on the /roi topic.
    
    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2011 Patrick Goebel.  All rights reserved.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
      
"""

import roslib
roslib.load_manifest('rbx_vision')
import rospy
import cv2
import cv2.cv as cv
import sys
from std_msgs.msg import String
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import time


class ROS2OpenCV2:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        
        rospy.on_shutdown(self.cleanup)
        
        self.node_name = node_name
        self.show_text = rospy.get_param("~show_text", True)
        self.show_features = rospy.get_param("~show_features", True)
        self.show_boxes = rospy.get_param("~show_boxes", True)
        self.flip_image = rospy.get_param("~flip_image", False)

        """ Initialize the Region of Interest and its publisher """
        self.ROI = RegionOfInterest()
        self.pubROI = rospy.Publisher("/roi", RegionOfInterest)
        
        """ Initialize a number of global variables """
        self.frame = None
        self.frame_size = None
        self.depth_image = None
        self.grey = None
        self.prev_grey = None
        self.selected_point = None
        self.selection = None
        self.drag_start = None
        self.keystroke = None
        self.detect_box = None
        self.track_box = None
        self.display_box = None
        self.keep_marker_history = False
        self.night_mode = False
        self.auto_face_tracking = False
        self.cps = 0 # Cylces per second = number of processing loops per second.
        self.cps_values = list()
        self.cps_n_values = 20
        self.busy = False

        """ Create the display window """
        self.cv_window_name = self.node_name
        cv.NamedWindow(self.cv_window_name, cv.CV_WINDOW_NORMAL)
        cv.ResizeWindow(self.cv_window_name, 640, 480)
        
        """ Create the cv_bridge object """
        self.bridge = CvBridge()
        
        """ Set a call back on mouse clicks on the image window """
        cv.SetMouseCallback (self.node_name, self.on_mouse_click, None)
        
        """ Subscribe to the raw camera image topic and set the image processing callback """
        self.image_sub = rospy.Subscriber("input_rgb_image", Image, self.image_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber("input_depth_image", Image, self.depth_callback, queue_size=1)
        
        rospy.loginfo("Starting " + self.node_name)
        
    def on_mouse_click(self, event, x, y, flags, param):
        """ We will usually use the mouse to select points to track or to draw a rectangle
            around a region of interest. """
        if not self.frame:
            return
        
        #if self.frame.origin:
            #y = self.frame.height - y
            
        if event == cv.CV_EVENT_LBUTTONDOWN and not self.drag_start:
            self.features = []
            self.track_box = None
            self.detect_box = None
            self.selected_point = (x, y)
            self.drag_start = (x, y)
            
        if event == cv.CV_EVENT_LBUTTONUP:
            self.drag_start = None
            self.classifier_initialized = False
            self.detect_box = self.selection
            
        if self.drag_start:
            xmin = max(0, min(x, self.drag_start[0]))
            ymin = max(0, min(y, self.drag_start[1]))
            xmax = min(self.frame.width, max(x, self.drag_start[0]))
            ymax = min(self.frame.height, max(y, self.drag_start[1]))
            self.selection = (xmin, ymin, xmax - xmin, ymax - ymin)
            
    def depth_callback(self, data):
        depth_image = self.convert_depth_image(data)
        
        if self.flip_image:    
            cv.Flip(depth_image)
            
        if not self.depth_image:
            (cols, rows) = cv.GetSize(depth_image)
            self.depth_image = cv.CreateMat(rows, cols, cv.CV_32FC1)
            
        cv.Copy(depth_image, self.depth_image)

    def image_callback(self, data):
        """ Store the image header in a global variable.  We need this so we can access
            the timestamp and frame_id of each frame. """
        self.image_header = data.header

        """ Time this loop to get cycles per second """
        start = time.time()
        
        """ Convert the raw image to OpenCV format using the convert_image() helper function """
        frame = self.convert_image(data)
        
        """ Some webcams invert the image """
        if self.flip_image:
            cv.Flip(frame)
                    
        """ Create a few images we will use for display """
        if not self.frame:
            self.frame_size = cv.GetSize(frame)
            self.cols, self.rows = cv.GetSize(frame)
            self.frame_type = frame.type
            self.frame = cv.CreateMat(self.rows, self.cols, self.frame_type)
            self.marker_image = cv.CreateMat(self.rows, self.cols, self.frame_type)
            self.display_image = cv.CreateMat(self.rows, self.cols, self.frame_type)
            self.processed_image = cv.CreateMat(self.rows, self.cols, self.frame_type)
            cv.SetZero(self.marker_image)
        
        """ Copy the current frame to the global image in case we need it elsewhere"""
        self.frame = cv.CloneMat(frame)
            
        if not self.keep_marker_history:
            cv.SetZero(self.marker_image)
        
        """ Process the image to detect and track objects or features """
        if not self.busy:
            processed_image = self.process_image(frame)
        else:
            time.sleep(0.0333)
            return
        
        """ If the result is a greyscale image, convert to 3-channel for display purposes """
        #if processed_image.channels == 1:
            #cv.CvtColor(processed_image, self.processed_image, cv.CV_GRAY2BGR)
        #else:
        self.processed_image = cv.CloneMat(processed_image)
        
        """ Display the user-selection rectangle or point."""
        self.display_markers()
        
        if self.night_mode:
            """ Night mode: only display the markers """
            cv.SetZero(self.processed_image)
            
        """ Merge the processed image and the marker image """ 
        cv.Or(self.processed_image, self.marker_image, self.display_image)

        if self.track_box is not None:
            try:
                (center, size, angle) = self.track_box
                pt1 = (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2))
                pt2 = (int(center[0] + size[0] / 2), int(center[1] + size[1] / 2))
                #cv.Rectangle(self.display_image, pt1, pt2, cv.RGB(255, 0, 0), 2, 8, 0)
                if self.show_boxes:
                    cv.EllipseBox(self.display_image, self.track_box, cv.CV_RGB(255, 0, 0), 1)
            except:
                x,y,w,h = self.track_box
                size = w, h
                center = x + w / 2, y + h / 2
                pt1 = (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2))
                pt2 = (int(center[0] + size[0] / 2), int(center[1] + size[1] / 2))
                try:
                    cv.Rectangle(self.display_image, pt1, pt2, cv.RGB(255, 0, 0), 1, 8, 0)
                except:
                    print pt1, pt2
        elif self.detect_box is not None:
            (pt1_x, pt1_y, w, h) = self.detect_box
            if self.show_boxes:
                cv.Rectangle(self.display_image, (pt1_x, pt1_y), (pt1_x + w, pt1_y + h), cv.RGB(255, 0, 0), 1, 8, 0)
        
        """ Handle keyboard events """
        self.keystroke = cv.WaitKey(5)
            
        end = time.time()
        duration = end - start
        fps = int(1.0 / duration)
        self.cps_values.append(fps)
        if len(self.cps_values) > self.cps_n_values:
            self.cps_values.pop(0)
        self.cps = int(sum(self.cps_values) / len(self.cps_values))
        
        if self.show_text:
            hscale = 0.2 * self.frame_size[0] / 160. + 0.1
            vscale = 0.2 * self.frame_size[1] / 120. + 0.1
            text_font = cv.InitFont(cv.CV_FONT_VECTOR0, hscale, vscale, 0, 1, 8)
            """ Print cycles per second (CPS) and resolution (RES) at top of the image """
            if self.frame_size[0] >= 640:
                vstart = 25
                voffset = int(50 + self.frame_size[1] / 120.)
            elif self.frame_size[0] == 320:
                vstart = 15
                voffset = int(35 + self.frame_size[1] / 120.)
            else:
                vstart = 10
                voffset = int(20 + self.frame_size[1] / 120.)
            cv.PutText(self.display_image, "CPS: " + str(self.cps), (10, vstart), text_font, cv.RGB(255, 255, 0))
            cv.PutText(self.display_image, "RES: " + str(self.frame_size[0]) + "X" + str(self.frame_size[1]), (10, voffset), text_font, cv.RGB(255, 255, 0))
        
        # Now display the image.
        cv.ShowImage(self.node_name, self.display_image)
        
        """ Process any keyboard commands """
        if 32 <= self.keystroke and self.keystroke < 128:
            cc = chr(self.keystroke).lower()
            if cc == 'n':
                self.night_mode = not self.night_mode
            elif cc == 'f':
                self.show_features = not self.show_features
            elif cc == 'b':
                self.show_boxes = not self.show_boxes
            elif cc == 't':
                self.show_text = not self.show_text
            elif cc == 'q':
                """ user has press the q key, so exit """
                rospy.signal_shutdown("User hit q key to quit.")      
          
    def convert_image(self, ros_image):
        try:
            frame = cv.GetMat(self.bridge.imgmsg_to_cv(ros_image, "bgr8"))
            return frame
        except CvBridgeError, e:
            print e
          
    def convert_depth_image(self, ros_image):
        try:
            depth_image = self.bridge.imgmsg_to_cv(ros_image, "32FC1")
            return depth_image
    
        except CvBridgeError, e:
            print e
          
    def process_image(self, frame): 
        # Since we aren't applying any filters in this base class, set the ROI to the selected region, if any.
        if not self.drag_start and not self.detect_box is None:         
            self.ROI = RegionOfInterest()
            self.ROI.x_offset = self.detect_box[0]
            self.ROI.y_offset = self.detect_box[1]
            self.ROI.width = self.detect_box[2]
            self.ROI.height = self.detect_box[3]
            
        self.pubROI.publish(self.ROI)
        
        return frame
    
    def display_markers(self):
        # If the user is selecting a region with the mouse, display the corresponding rectangle for feedback.
        if self.drag_start and self.is_rect_nonzero(self.selection):
            x,y,w,h = self.selection
            cv.Rectangle(self.marker_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            self.selected_point = None

        # Else if the user has clicked on a point on the image, display it as a small circle.            
        elif not self.selected_point is None:
            x = self.selected_point[0]
            y = self.selected_point[1]
            cv.Circle(self.marker_image, (x, y), 3, (0, 255, 255), 2)
        
    def is_rect_nonzero(self, r):
        # First assume a simple CvRect type
        try:
            (_, _, w, h) = r
            return (w > 0) and (h > 0)
        except:
            # Otherwise, assume a CvBox2D type
            ((_,_), (w,h), a) = r
            return (w > 0) and (h > 0)
        
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
        
    def cleanup(self):
        print "Shutting down vision node."
        cv2.destroyAllWindows()       

def main(args):
    # Display a help message if appropriate.
    help_message = ""
          
    print help_message
          
    # Fire up the node.
    ROS2OpenCV2("ros2opencv")
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ros2opencv node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
