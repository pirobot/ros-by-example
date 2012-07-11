#!/usr/bin/env python

""" nav_square.py - Version 0.1 2012-03-24

    A basic demo of the using odometry data to move the robot
    along a square trajectory.

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
import PyKDL
from math import radians, copysign, sqrt, pow, pi
import threading

class NavSquare():
    def __init__(self):
        # Give the node a name
        rospy.init_node('nav_square', anonymous=False)
        
        # Set rospy to exectute a shutdown function when terminating the script
        rospy.on_shutdown(self.shutdown)
        
        # Create a lock for reading odometry values
        self.lock = threading.Lock()
                
        # How fast will we check the odometry values?
        rate = 30
        r = rospy.Rate(rate)
        
        # Set the parameters for the target square
        goal_distance = rospy.get_param("~goal_distance", 1.0)      # meters
        goal_angle = rospy.get_param("~goal_angle", radians(90))    # degrees converted to radians
        linear_speed = rospy.get_param("~linear_speed", 0.2)        # meters per second
        angular_speed = rospy.get_param("~angular_speed", 0.7)      # radians per second
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist)
        
        # Variable to hold the current odometry values
        self.odom = Odometry()
        
        # Subscribe to the /odom topic and set the callback
        rospy.Subscriber('/odom', Odometry, self.update_odom)
        
        # Wait for the /odom topic to become available
        rospy.wait_for_message('/odom', Odometry)
        
        # Wait until we actually have some data
        while self.odom == Odometry():
            rospy.sleep(1)

        # Cycle through the four sides of the square
        for i in range(4):
            # Initialize the movement command
            move_cmd = Twist()
            
            # Set the movement command to forward motion
            move_cmd.linear.x = linear_speed
            
            # Get the starting position values     
            x_start = self.odom.pose.pose.position.x
            y_start = self.odom.pose.pose.position.y
            
            # Keep track of the distance traveled
            distance = 0
            
            # Enter the loop to move along a side
            while distance < goal_distance and not rospy.is_shutdown():
                # Publish the Twist message and sleep 1 cycle         
                self.cmd_vel.publish(move_cmd)
                r.sleep()
        
                # Compute the Euclidean distance from the start
                with self.lock:
                    distance = sqrt(pow((self.odom.pose.pose.position.x - x_start), 2)
                                  + pow((self.odom.pose.pose.position.y - y_start), 2))

            # Stop the robot before rotating
            move_cmd = Twist()
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(1.0)
            
            # Set the movement command to a rotation
            move_cmd.angular.z = angular_speed
            
            # Track the last angle measured
            last_angle = self.odom_angle
            
            # Track how far we have turned
            turn_angle = 0
            
            # Begin the rotation
            while turn_angle < goal_angle and not rospy.is_shutdown():
                # Publish the Twist message and sleep 1 cycle         
                self.cmd_vel.publish(move_cmd)
                r.sleep()
                                
                with self.lock:
                    delta_angle = normalize_angle(self.odom_angle - last_angle)
                
                turn_angle += delta_angle
                last_angle = self.odom_angle

            move_cmd = Twist()
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(1.0)
            
        # Stop the robot when we are done
        self.cmd_vel.publish(Twist())
        
    def update_odom(self, msg):
        with self.lock:
            self.odom = msg                            
            q = msg.pose.pose.orientation
            self.odom_angle = quat_to_angle(q)
            
    def shutdown(self):
        # Always stop the robot when shutting down the node
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

def quat_to_angle(quat):
        rot = PyKDL.Rotation.Quaternion(quat.x, quat.y, quat.z, quat.w)
        return rot.GetRPY()[2]
            
def normalize_angle(angle):
        res = angle
        while res > pi:
            res -= 2.0 * pi
        while res < -pi:
            res += 2.0 * pi
        return res
        
if __name__ == '__main__':
    try:
        NavSquare()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation terminated.")

