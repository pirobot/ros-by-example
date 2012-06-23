#!/usr/bin/env python

"""
    object_tracker.py - Version 1.0 2012-06-01
    
    Rotate the robot left or right to follow a target published on the /roi topic.
    
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

import roslib; roslib.load_manifest('rbx_apps')
import rospy
from sensor_msgs.msg import RegionOfInterest, CameraInfo
from geometry_msgs.msg import Twist
from math import copysign

class ObjectTracker():
    def __init__(self):
        rospy.init_node("object_tracker")
                
        # Set the shutdown function (stop the robot)
        rospy.on_shutdown(self.shutdown)
        
        # The maximum rotation speed in radians per second
        self.max_rotation_speed = rospy.get_param("~max_rotation_speed", 2.0)
        
        # The minimum rotation speed in radians per second
        self.min_rotation_speed = rospy.get_param("~min_rotation_speed", 0.5)
        
        # Sensitivity to target displacements.  Setting this too high
        # can lead to oscillations of the robot.
        self.gain = rospy.get_param("~gain", 2.0)
        
        # How often should we update our response to object motion?
        self.rate = rospy.get_param("~rate", 10)
        r = rospy.Rate(self.rate) 
        
        # The pan threshold (% of image width) indicates how far off-center the ROI needs to be before we react
        self.angular_threshold = rospy.get_param("~angular_threshold", 0.1)

        # Publisher to control the robot's movement
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist)
        
        # Intialize the movement command
        self.move_cmd = Twist()
        
        # We will get the image width and height from the camera_info topic
        self.image_width = 0
        self.image_height = 0
        
        # Wait for the camera_info topic to become available
        rospy.loginfo("Waiting for camera_info topic...")
        rospy.wait_for_message('camera_info', CameraInfo)
        
        # Subscribe the camera_info topic to get the image width and height
        rospy.Subscriber('camera_info', CameraInfo, self.get_camera_info)

        # Wait until we actually have the camera data
        while self.image_width == 0 or self.image_height == 0:
            rospy.sleep(1)
            
        rospy.loginfo("Image size: " + str(self.image_width) + " x " + str(self.image_height))
        
        # Subscribe to the ROI topic and set the callback to update the robot's motion
        rospy.Subscriber('roi', RegionOfInterest, self.set_cmd_vel)
        
        # Wait until we have an ROI to follow
        rospy.wait_for_message('roi', RegionOfInterest)
        
        # Begin the tracking loop
        while not rospy.is_shutdown():
            # Send the latest Twist command to the robot
            self.cmd_vel_pub.publish(self.move_cmd)
            
            # Sleep for 1/self.rate seconds
            r.sleep()
    
    def set_cmd_vel(self, msg):
        # If the ROI was lost (msg.width=0), stop the robot
        if msg.width == 0:
            self.move_cmd = Twist()
            return
        
        # Compute the center of the ROI based on the x_offset and image width
        angular_offset = msg.x_offset + (msg.width / 2.0) - (self.image_width / 2.0)
        
        try:
            percent_offset = abs(float(angular_offset) / float(self.image_width))
        except:
            percent_offset = 0
                              
        # Pan the camera only if the displacement of the COG exceeds the threshold
        if percent_offset > self.angular_threshold:
            # Set the rotation speed proportional to the displacement of the target
            try:
                speed = self.gain * angular_offset / (self.image_width / 2.0)
                if speed < 0:
                    direction = -1
                else:
                    direction = 1
                self.move_cmd.angular.z = -direction * max(self.min_rotation_speed, min(self.max_rotation_speed, abs(speed)))
            except:
                self.move_cmd = Twist()
        else:
            # Otherwise stop the robot
            self.move_cmd = Twist()
            
    def get_camera_info(self, msg):
        self.image_width = msg.width
        self.image_height = msg.height
        
    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)     
                   
if __name__ == '__main__':
    try:
        ObjectTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Object tracking node terminated.")




