#!/usr/bin/env python

""" out_and_back.py - Version 0.1 2012-03-24

    A basic demo of the ROS Twist message to control a differential drive robot.

    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2012 Patrick Goebel.  All rights reserved.

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

import roslib; roslib.load_manifest('rbx_nav')
import rospy
from geometry_msgs.msg import Twist
from time import sleep
from math import pi

class OutAndBack():
    def __init__(self):
        # Give the node a name
        rospy.init_node('out_and_back', anonymous=False)

        # Set rospy to exectute a shutdown function when exiting       
        rospy.on_shutdown(self.shutdown)
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist)
        
        # How fast will we update the robot's movement?
        rate = 30
        r = rospy.Rate(rate)
        
        # Initialize the forward movement command
        move_forward = Twist()
        
        # Set the forward linear speed to 0.2 meters per second 
        move_forward.linear.x = 0.2
        
        # Initialize the rotate command
        rotate_left = Twist()
        
        # Set the rotation speed to 1.0 radians per second
        rotate_left.angular.z = 1.0
        
        i = 0
        # Loop once for each leg of the trip
        while i < 2 and not rospy.is_shutdown():
            # Move forward for 5 seconds (approx 1 meter)
            ticks = int(5 * rate)
            for t in range(ticks):
                self.cmd_vel.publish(move_forward)
                r.sleep()
    
            # Rotate left roughly 180 degrees           
            ticks = int(pi * rate)
            for t in range(ticks):           
                self.cmd_vel.publish(rotate_left)
                r.sleep()
            
            i += 1

        # Stop the robot.
        self.cmd_vel.publish(Twist())
        
    def shutdown(self):
        # Always stop the robot when shutting down the node.
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
 
if __name__ == '__main__':
    try:
        OutAndBack()
    except rospy.ROSInterruptException:
        rospy.loginfo("Out-and-Back node terminated.")

