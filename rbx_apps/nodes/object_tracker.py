#!/usr/bin/env python

"""
    object_tracker.py - Version 1.0 2012-06-01
    
    Follow a target published on the /roi topic.
    
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

class ObjectTracker():
    def __init__(self):
        rospy.init_node("object_tracker")
        
        rospy.on_shutdown(self.shutdown)
        
        # What is our max rotation speed in radians per second?
        self.max_rotation_speed = rospy.get_param("~max_rotation_speed", 2.0)
        
        # How quickly should we respond to target displacements?  Setting this too high
        # can lead to oscillations of the robot.
        self.k_angular = rospy.get_param("~k_angular", 1.5)
        
        # How often should we update our response to object motion?
        self.rate = rospy.get_param("~rate", 10)
        r = rospy.Rate(self.rate) 
        
        # The pan threshold indicates how far off-center the ROI needs to be before we react
        self.angular_threshold = int(rospy.get_param("~angular_threshold", 20))

        # Publisher to control the robot's movement
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist)
        
        self.move_cmd = Twist()
        
        self.tracking_seq = 0
        self.last_tracking_seq = -1
        
        self.image_width = 0
        self.image_height = 0
        
        rospy.Subscriber('roi', RegionOfInterest, self.setCmdVel)
        rospy.Subscriber('camera_info', CameraInfo, self.getCameraInfo)
        
        # Wait for the camera_info topic to become available
        rospy.wait_for_message('camera_info', CameraInfo)

        # Wait until we actually have the camera data
        while self.image_width == 0 or self.image_height == 0:
            rospy.sleep(1)
        
        while not rospy.is_shutdown():
            # Update the robot's motion depending on the target's location
            if self.last_tracking_seq == self.tracking_seq:
                self.move_cmd = Twist()
            else:
                self.last_tracking_seq = self.tracking_seq
                
            self.cmd_vel_pub.publish(self.move_cmd)
            r.sleep()
    
    def setCmdVel(self, msg):
        # When OpenCV loses the ROI, the message stops updating.  Use this counter to
        # determine when it stops. 
        self.tracking_seq += 1

        # Compute the center of the ROI
        angular_offset = msg.x_offset + msg.width / 2 - self.image_width / 2
                  
        # Pan the camera only if the displacement of the COG exceeds the threshold
        if abs(angular_offset) > self.angular_threshold:
            # Set the rotation speed proportional to the displacement of the horizontal displacement
            # of the target
            try:
                self.move_cmd.angular.z = -min(self.max_rotation_speed, self.k_angular * angular_offset / float(self.image_width))
            except:
                self.move_cmd = Twist()
        else:
            self.move_cmd = Twist()
            
    def getCameraInfo(self, msg):
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




