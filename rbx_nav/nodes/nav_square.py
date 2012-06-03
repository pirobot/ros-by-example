#!/usr/bin/env python

""" nav_square.py - Version 0.1 2012-03-24

    A basic demo of the using odometry data to move the robot along a square trajectory.

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
roslib.load_manifest('rbx_nav')

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import radians, copysign, sqrt, pow, pi

class NavSquare():
    def __init__(self):
        rospy.init_node('nav_square', anonymous=False)
        
        rospy.on_shutdown(self.shutdown)
        
        rate = 100
        tick = 1.0 / rate
        
        square_size = 1.0 # meters
        turn_angle = radians(90) # degrees
        speed_linear = 0.5
        speed_angular = 1.0
        tolerance_linear = 0.025 # meters
        tolerance_angular = radians(2.5) # degrees
        print tolerance_angular
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist)
        
        self.odom = Odometry()
        
        # Subscribe to the /odom topic to get odometry data
        rospy.Subscriber('/odom', Odometry, self.update_odom)
        
        # Wait for the /odom topic to become available
        rospy.wait_for_message('/odom', Odometry)
        
        # Wait until we actually have some data
        while self.odom == Odometry():
            rospy.sleep(1)

        # Cycle through the four sides of the square
        for i in range(4):
            # Get the starting odometry values     
            odom_start = self.odom
            x_start = odom_start.pose.pose.position.x
            y_start = odom_start.pose.pose.position.y
            
            # First move along a side
            move_cmd = Twist()
            waypoint_success = False
            
            while not waypoint_success and not rospy.is_shutdown():
                error = sqrt(pow((x_start - self.odom.pose.pose.position.x), 2) +  pow((y_start - self.odom.pose.pose.position.y), 2)) - square_size
                if abs(error) <  tolerance_linear:
                    waypoint_success = True
                else:
                    move_cmd.linear.x = copysign(speed_linear, -1 * error)                
                    self.cmd_vel.publish(move_cmd)
                    rospy.sleep(tick)

            # Stop the robot before rotating
            self.cmd_vel.publish(Twist())
            waypoint_success = False
            
            # Now rotate 90 degrees
            move_cmd = Twist()
            orientation = euler_from_quaternion([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w], axes='sxyz')
            start_angle = orientation[2]
            if start_angle < 0:
                start_angle = 2 * pi + start_angle
            
            #rospy.loginfo("Start Angle: " + str(start_angle))

            while not waypoint_success and not rospy.is_shutdown():
                orientation = euler_from_quaternion([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w], axes='sxyz')
                
                current_angle = orientation[2]
                if current_angle < 0:
                    current_angle = 2 * pi + current_angle
                                    
                target_angle = (start_angle + turn_angle) % (2 * pi)
                
                error = current_angle -  target_angle
                
                if abs(error) >= pi:
                    error = abs(error) - 2 * pi

                if abs(error) <  tolerance_angular:
                    waypoint_success = True
                else:
                    move_cmd.angular.z = copysign(speed_angular, -1 * error)                
                    self.cmd_vel.publish(move_cmd)
                    rospy.sleep(tick)

        # Stop the robot.
        self.cmd_vel.publish(Twist())
        
    def update_odom(self, msg):
        self.odom = msg
        
    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
 
if __name__ == '__main__':
    try:
        NavSquare()
    except rospy.ROSInterruptException:
        rospy.loginfo("Dead reckoning finished.")
