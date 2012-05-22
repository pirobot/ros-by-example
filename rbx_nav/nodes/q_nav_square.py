#! /usr/bin/python

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
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply, vector_norm
from time import sleep
from math import radians, copysign, sqrt, pow, pi
import numpy as np

class NavSquare():
    def __init__(self):
        rospy.init_node('nav_square', anonymous=False)
        
        rospy.on_shutdown(self.shutdown)
        
        rate = 30
        tick = 1.0 / rate
        
        square_size = 0.2 # meters
        turn_angle = radians(90) # degrees
        q_turn_angle = Quaternion()
        q_turn_angle = quaternion_from_euler(0, 0, turn_angle, axes='sxyz')
        speed_linear = 0.2
        speed_angular = 0.5
        tolerance_linear = 0.05 # meters
        tolerance_angular = radians(5)
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist)
        
        self.odom = Odometry()
        
        # Subscribe to the /odom topic to get odometry data
        odom_sub = rospy.Subscriber('/odom', Odometry, self.update_odom)
        
        # Wait for the /odom topic to become available
        rospy.wait_for_message('/odom', Odometry)
        
        # Wait until we actually have some data
        while self.odom == Odometry():
            rospy.sleep(1)
            
        for i in range(4):
            # Get the starting odometry values     
            odom_start = self.odom
            x_start = odom_start.pose.pose.position.x
            y_start = odom_start.pose.pose.position.y
            q_start = [self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]

            # First move along a side
            move_cmd = Twist()
            move_cmd.linear.x = speed_linear
            waypoint_success = False
            
            while not waypoint_success:
                error = sqrt(pow((x_start - self.odom.pose.pose.position.x), 2) +  pow((y_start - self.odom.pose.pose.position.y), 2)) - square_size
                if abs(error) <  tolerance_linear:
                    waypoint_success = True
                else:
                    self.cmd_vel.publish(move_cmd)
                    rospy.sleep(tick)

            # Stop the robot before rotating
            self.cmd_vel.publish(Twist())
            waypoint_success = False
            
            # Now rotate 90 degrees
            move_cmd = Twist()
            while not waypoint_success: 
                q_current = [self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]
                
                q_error = q_current - quaternion_multiply(q_turn_angle, q_start)
                q_error = q_error / vector_norm(q_error)
                euler_angles = euler_from_quaternion(q_error, axes='sxyz')
                rospy.loginfo(euler_angles)
                yaw_error = euler_angles[2]
                
                #rospy.loginfo(yaw_error)

                if abs(yaw_error) <  tolerance_angular:
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
