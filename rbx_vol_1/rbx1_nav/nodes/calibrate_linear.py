#!/usr/bin/env python

""" calibrate_linear.py - Version 0.1 2012-03-24

    Move the robot 1.0 meter to check on the PID parameters of the base controller.

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

import roslib; roslib.load_manifest('rbx1_nav')
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import copysign, sqrt, pow

class CalibrateLinear():
    def __init__(self):
        # Give the node a name
        rospy.init_node('calibrate_linear', anonymous=False)
        
        # Set rospy to exectute a shutdown function when terminating the script
        rospy.on_shutdown(self.shutdown)
        
        # How fast will we check the odometry values?
        rate = 50
        tick = 1.0 / rate
        
        # Set the distance to travel
        distance = 1.0 # meters
        speed_linear = 0.15 # meters per second
        tolerance_linear = 0.01 # meters
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist)
        
        # Variable to hold the current odometry values
        self.odom = Odometry()
        
        # Subscribe to the /odom topic to get odometry data.  Set the callback to the self.odom_update function.
        rospy.Subscriber('/odom', Odometry, self.update_odom)
        
        # Wait for the /odom topic to become available
        rospy.wait_for_message('/odom', Odometry)
        
        # Wait until we actually have some data
        while self.odom == Odometry():
            rospy.sleep(1)
  
        odom_start = self.odom
        x_start = odom_start.pose.pose.position.x
        y_start = odom_start.pose.pose.position.y
            
        move_cmd = Twist()
        target_reached = False
            
        while not target_reached and not rospy.is_shutdown():
            # Compute the Euclidean distance from the target point
            error = sqrt(pow((x_start - self.odom.pose.pose.position.x), 2) +  pow((y_start - self.odom.pose.pose.position.y), 2)) - distance
                
            # Are we close enough?
            if abs(error) <  tolerance_linear:
                target_reached = True
            else:
                # If not, move in the appropriate direction
                move_cmd.linear.x = copysign(speed_linear, -1 * error)                
                self.cmd_vel.publish(move_cmd)
                rospy.sleep(tick)

        # Stop the robot
        move_cmd = Twist()
        self.cmd_vel.publish(move_cmd)
        
    def update_odom(self, msg):
        self.odom = msg
        
    def shutdown(self):
        # Always stop the robot when shutting down the node
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
 
if __name__ == '__main__':
    try:
        CalibrateLinear()
    except rospy.ROSInterruptException:
        rospy.loginfo("Calibration terminated.")
