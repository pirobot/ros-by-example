#! /usr/bin/python

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

import roslib
roslib.load_manifest('pi_nav')

import rospy
from geometry_msgs.msg import Twist
from time import sleep

class OutAndBack():
    def __init__(self):
        rospy.init_node('out_and_back', anonymous=False)
        
        rospy.on_shutdown(self.shutdown)
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist)
        
        rate = 30
        tick = 1.0 / rate
        
        move_forward = Twist()
        move_forward.linear.x = 0.2
        
        rotate_left = Twist()
        rotate_left.angular.z = 1.0
        
        for i in range(2):
            # Move forward for 3 seconds
            ticks = int(3 * rate)
            for t in range(ticks):
                self.cmd_vel.publish(move_forward)
                rospy.sleep(tick)
    
            # Rotate left roughly 180 degrees           
            ticks = int(3.07 * rate)
            for t in range(ticks):           
                self.cmd_vel.publish(rotate_left)
                rospy.sleep(tick)

        # Stop the robot.
        self.cmd_vel.publish(Twist())
        
    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
 
if __name__ == '__main__':
    try:
        OutAndBack()
    except rospy.ROSInterruptException:
        rospy.loginfo("Twist test finished.")
