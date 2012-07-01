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

import roslib; roslib.load_manifest('rbx_nav')
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import radians, copysign, sqrt, pow, pi

class NavSquare():
    def __init__(self):
        # Give the node a name
        rospy.init_node('nav_square', anonymous=False)
        
        # Set rospy to exectute a shutdown function when terminating the script
        rospy.on_shutdown(self.shutdown)
        
        # How fast will we check the odometry values?
        rate = 50
        r = rospy.Rate(rate)
        
        # Set the parameters for the target square
        square_size = rospy.get_param("~square_size", 1.0) # meters
        turn_angle = rospy.get_param("~turn_angle", radians(90)) # degrees converted to radians
        speed_linear = rospy.get_param("~speed_linear", 0.2) # meters per second
        speed_angular = rospy.get_param("~speed_angular", 0.7) # radians per second
        tolerance_linear = rospy.get_param("~tolerance_linear", 0.02) # meters
        tolerance_angular = rospy.get_param("~tolerance_angular", radians(2)) # degrees converted to radians
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist)
        
        # Variable to hold the current odometry values
        self.odom = Odometry()
        
        # Subscribe to the /odom topic to get odometry data.
        # Set the callback to the self.odom_update function.
        rospy.Subscriber('/odom', Odometry, self.update_odom)
        
        # Wait for the /odom topic to become available
        rospy.wait_for_message('/odom', Odometry)
        
        # Wait until we actually have some data
        while self.odom == Odometry():
            rospy.sleep(1)

        """ Cycle through the four sides of the square """
        for i in range(4):
            # Get the starting odometry values     
            odom_start = self.odom
            x_start = odom_start.pose.pose.position.x
            y_start = odom_start.pose.pose.position.y
            
            """ First move along a side """
            # Initialize the movement command
            move_cmd = Twist()
            
            # A flag indicating when we have reached a waypoint
            waypoint_success = False
            
            # Enter the loop to move along a side
            while not waypoint_success and not rospy.is_shutdown():
                # Compute the Euclidean distance from the start
                distance = sqrt(pow((x_start - self.odom.pose.pose.position.x), 2)
                            + pow((y_start - self.odom.pose.pose.position.y), 2))
                
                # How close are we to the goal?
                error = distance - square_size
                
                # Are we close enough?
                if abs(error) <  tolerance_linear:
                    waypoint_success = True
                else:
                    # If not, move in the appropriate direction
                    move_cmd.linear.x = copysign(speed_linear, -1 * error)                
                    self.cmd_vel.publish(move_cmd)
                    r.sleep()

            # Stop the robot before rotating
            move_cmd = Twist()
            self.cmd_vel.publish(move_cmd)
            waypoint_success = False
            
            """ Now rotate 90 degrees """
            # Get the starting orientation as a quaternion
            q = [self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w]
            
            # Convert the quaternion to Euler angles
            euler_angles = euler_from_quaternion(q, axes='sxyz')
            
            # The third value (index 2) is the rotation about the z axis
            start_angle = euler_angles[2]
            
            # The Euler angles jump from 180 to -180 so we
            # have to do some adjustment to compensate
            if start_angle < 0:
                start_angle = 2 * pi + start_angle
            
            # Set the target angle modulo 360 degrees to handle wrap around
            target_angle = (start_angle + turn_angle) % (2 * pi)
            
            # Do the rotation
            while not waypoint_success and not rospy.is_shutdown():
                # Get the current orientation as a quaternion
                q = [self.odom.pose.pose.orientation.x,
                    self.odom.pose.pose.orientation.y,
                    self.odom.pose.pose.orientation.z,
                    self.odom.pose.pose.orientation.w]
                
                # Convert to Euler angles
                euler_angles = euler_from_quaternion(q, axes='sxyz')
                
                # Get the rotation around the z-axis
                current_angle = euler_angles[2]
                
                # The Euler angles jump from 180 to -180 so we have to
                # do some adjustment to compensate
                if current_angle < 0:
                    current_angle = 2 * pi + current_angle
                
                # How far away are we from the target angle?
                error = current_angle -  target_angle
                
                # Adjust for the circular nature of angle arithmetic
                if abs(error) >= pi:
                    error = abs(error) - 2 * pi

                # Are we close enough?
                if abs(error) <  tolerance_angular:
                    waypoint_success = True
                else:
                    # If not, rotate in the appropriate direction
                    move_cmd.angular.z = copysign(speed_angular, -1 * error)                
                    self.cmd_vel.publish(move_cmd)
                    r.sleep()

        # Stop the robot when we are done
        self.cmd_vel.publish(Twist())
        
    def update_odom(self, msg):
        self.odom = msg
        
    def shutdown(self):
        # Always stop the robot when shutting down the node
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
 
if __name__ == '__main__':
    try:
        NavSquare()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation terminated.")

