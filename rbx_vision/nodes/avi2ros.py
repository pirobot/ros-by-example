#!/usr/bin/env python

""" avi2ros.py - Version 0.1 2011-04-28

    Read in an AVI video file and republish as a ROS Image topic.
    
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
roslib.load_manifest('pi_video_tracker')
import sys
import rospy
import cv
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class AVI2ROS:
    def __init__(self):
        rospy.init_node('avi2ros', anonymous=True)
        
        self.input = rospy.get_param("~input", "")
        image_pub = rospy.Publisher("output", Image)
        
        self.fps = rospy.get_param("~fps", 25)
        self.loop = rospy.get_param("~loop", False)
        self.start_paused = rospy.get_param("~start_paused", False)
        self.show_text = True
        
        rospy.on_shutdown(self.cleanup)
        
        video = cv.CaptureFromFile(self.input)
        fps = int(cv.GetCaptureProperty(video, cv.CV_CAP_PROP_FPS))
        
        """ Bring the fps up to the specified rate """
        try:
            fps = int(fps * self.fps / fps)
        except:
            fps = self.fps
    
        cv.NamedWindow("AVI Video", True) # autosize the display
        cv.MoveWindow("AVI Video", 650, 100)

        bridge = CvBridge()
                
        self.paused = self.start_paused
        self.keystroke = None
        self.restart = False
        
        # Get the first frame to display if we are starting in the paused state.
        frame = cv.QueryFrame(video)
        image_size = cv.GetSize(frame)
        
        text_frame = cv.CloneImage(frame)
        cv.Zero(text_frame)
    
        while not rospy.is_shutdown():
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
                
            if self.restart:
                video = cv.CaptureFromFile(self.input)
                self.restart = None
    
            if not self.paused:
                frame = cv.QueryFrame(video)
                
            if frame == None:
                if self.loop:
                    self.restart = True
            else:
                if self.show_text:
                    text_font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.2, 1, 0, 1, 8)
                    cv.PutText(text_frame, "Keyboard commands:", (20, int(image_size[1] * 0.6)), text_font, cv.RGB(255, 255, 0))
                    cv.PutText(text_frame, " ", (20, int(image_size[1] * 0.65)), text_font, cv.RGB(255, 255, 0))
                    cv.PutText(text_frame, "space - toggle pause/play", (20, int(image_size[1] * 0.72)), text_font, cv.RGB(255, 255, 0))
                    cv.PutText(text_frame, "     r - restart video from beginning", (20, int(image_size[1] * 0.79)), text_font, cv.RGB(255, 255, 0))
                    cv.PutText(text_frame, "     t - hide/show this text", (20, int(image_size[1] * 0.86)), text_font, cv.RGB(255, 255, 0))
                    cv.PutText(text_frame, "     q - quit the program", (20, int(image_size[1] * 0.93)), text_font, cv.RGB(255, 255, 0))
                
                cv.Add(frame, text_frame, text_frame)
                cv.ShowImage("AVI Video", text_frame)
                cv.Zero(text_frame)
                
                try:
                    image_pub.publish(bridge.cv_to_imgmsg(frame, "bgr8"))
                except CvBridgeError, e:
                    print e         
    
    def cleanup(self):
            print "Shutting down vision node."
            cv.DestroyAllWindows()

def main(args):
    help_message =  "Hot keys: \n" \
          "\tq     - quit the program\n" \
          "\tr     - restart video from beginning\n" \
          "\tspace - toggle pause/play\n"

    print help_message
    
    try:
        a2r = AVI2ROS()
    except KeyboardInterrupt:
        print "Shutting down avi2ros..."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
