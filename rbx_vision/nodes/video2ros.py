#!/usr/bin/env python

""" video2ros.py - Version 0.1 2012-05-31

    Read in recorded video file and republish as a ROS Image topic.
    
    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2012 Patrick Goebel.  All rights reserved.

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
import sys
import rospy
from cv2 import cv as cv
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class VIDEO2ROS:
    def __init__(self):
        rospy.init_node('video2ros', anonymous=False)
        
        self.input = rospy.get_param("~input", "")
        image_pub = rospy.Publisher("output", Image)
        
        self.fps = rospy.get_param("~fps", 25)
        self.loop = rospy.get_param("~loop", False)
        self.start_paused = rospy.get_param("~start_paused", False)
        self.show_text = True
        self.resize = False
        self.last_frame = None
        
        rospy.on_shutdown(self.cleanup)
        
        self.capture = cv2.VideoCapture(self.input)
        fps = self.capture.get(cv.CV_CAP_PROP_FPS)
        
        self.frame_size = (self.capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT), self.capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))
        
        if fps == 0.0:
            print "Video source", self.input, "not found!"
            return None        
        
        """ Bring the fps up to the specified rate """
        try:
            fps = int(fps * self.fps / fps)
        except:
            fps = self.fps
    
        cv.NamedWindow("Video Playback", True) # autosize the display
        cv.MoveWindow("Video Playback", 650, 100)

        bridge = CvBridge()
                
        self.paused = self.start_paused
        self.keystroke = None
        self.restart = False
    
        while not rospy.is_shutdown():
            frame = self.get_frame()
                
            try:
                image_pub.publish(bridge.cv_to_imgmsg(cv.fromarray(frame), "bgr8"))
            except CvBridgeError, e:
                print e
            
            display_image = frame.copy()
            
            if self.show_text:
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(display_image, "Keyboard commands:", (20, int(self.frame_size[0] * 0.6)), font_face, font_scale, cv.RGB(255, 255, 0))
                cv2.putText(display_image, " ", (20, int(self.frame_size[0] * 0.65)), font_face, font_scale, cv.RGB(255, 255, 0))
                cv2.putText(display_image, "space - toggle pause/play", (20, int(self.frame_size[0] * 0.72)), font_face, font_scale, cv.RGB(255, 255, 0))
                cv2.putText(display_image, "     r - restart video from beginning", (20, int(self.frame_size[0] * 0.79)), font_face, font_scale, cv.RGB(255, 255, 0))
                cv2.putText(display_image, "     t - hide/show this text", (20, int(self.frame_size[0] * 0.86)), font_face, font_scale, cv.RGB(255, 255, 0))
                cv2.putText(display_image, "     q - quit the program", (20, int(self.frame_size[0] * 0.93)), font_face, font_scale, cv.RGB(255, 255, 0))
            
            # Merge the processed image and the marker image
            display_image = cv2.bitwise_or(frame, display_image)
            
            # Now display the image.
            cv2.imshow("Video Playback", display_image)
                    
            """ Handle keyboard events """
            self.keystroke = cv.WaitKey(1000 / fps)

            """ Process any keyboard commands """
            if 32 <= self.keystroke and self.keystroke < 128:
                cc = chr(self.keystroke).lower()
                if cc == 'q':
                    """ user has press the q key, so exit """
                    rospy.signal_shutdown("User hit q key to quit.")
                elif cc == ' ':
                    """ Pause or continue the video """
                    self.paused = not self.paused
                elif cc == 'r':
                    """ Restart the video from the beginning """
                    self.restart = True
                elif cc == 't':
                    """ Toggle display of text help message """
                    self.show_text = not self.show_text        
        
                    
    def get_frame(self):
        if self.paused and not self.last_frame is None:
            frame = self.last_frame
        else:
            ret, frame = self.capture.read()

        if frame is None:
            if self.loop_video:
                self.restart = True     
                
        if self.restart:
            self.capture = cv2.VideoCapture(self.input)
            self.restart = False
            ret, frame = self.capture.read()
            
        if self.resize:
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.last_frame = frame
            
        return frame
    
    def cleanup(self):
            print "Shutting down video2ros node."
            cv2.destroyAllWindows()

def main(args):
    help_message =  "Hot keys: \n" \
          "\tq     - quit the program\n" \
          "\tr     - restart video from beginning\n" \
          "\tspace - toggle pause/play\n"

    print help_message
    
    try:
        v2r = VIDEO2ROS()
    except KeyboardInterrupt:
        print "Shutting down video2ros..."
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
