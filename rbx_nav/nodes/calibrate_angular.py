#!/usr/bin/env python

""" calibrate_angular.py - Version 0.1 2012-03-24

    Rotate the robot 360 degrees to check the PID parameters of the base controller.

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
from tf.transformations import quaternion_from_euler

from math import radians, copysign, pi
import PyKDL
import threading

class CalibrateAngular():
    def __init__(self):
        self.lock = threading.Lock()                                            

        # Give the node a name
        rospy.init_node('calibrate_angular', anonymous=False)
        
        # Set rospy to exectute a shutdown function when terminating the script
        rospy.on_shutdown(self.shutdown)
        
        # How fast will we check the odometry values?
        rate = 50
        tick = 1.0 / rate
        
        # Set the parameters of our target square
        turn_angle = quaternion_from_euler(0, 180, 0) # degrees converted to radians
        q_turn_angle = PyKDL.Rotation.Quaternion(*turn_angle)
        speed_angular = 0.7 # radians per second
        tolerance_angular = radians(5) # degrees converted to radians
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist)
        
        # Variable to hold the current odometry angle in radians
        q_start_angle = None
        self.q_odom_angle = None
        
        # Subscribe to the /odom topic to get odometry data.  Set the callback to the self.odom_update function.
        rospy.Subscriber('/odom', Odometry, self.update_odom)
        
        # Wait for the /odom topic to become available
        rospy.wait_for_message('/odom', Odometry)
        
        self.odom = Odometry()
        
        # Wait until we actually have some data
        while self.q_odom_angle is None:
            rospy.sleep(1)

        # Get the starting angle
        q_start_angle = self.q_odom_angle
        
        # Set the target angle   
        q_target_angle = q_turn_angle * q_start_angle 
        
        # Are we there yet?
        target_reached = False
        
        while not target_reached and not rospy.is_shutdown():
            move_cmd = Twist()
            
            # Get the angular distance from the target
            with self.lock:    
                delta_q = self.q_odom_angle * q_target_angle.Inverse()
            
            delta_rpy = delta_q.GetRPY()
            delta_angle = delta_rpy[2]
                
            # Are we close enough?
            if abs(delta_angle) <  tolerance_angular:
                target_reached = True
            else:
                # If not, rotate in the appropriate direction
                move_cmd.angular.z = copysign(speed_angular, -1 * delta_angle)                
                self.cmd_vel.publish(move_cmd)
                rospy.sleep(tick)

        # Stop the robot
        self.cmd_vel.publish(Twist())
        
    def quat_to_angle(self, quat):                                                        
        rot = PyKDL.Rotation.Quaternion(quat.x, quat.y, quat.z, quat.w)      
        return rot.GetRPY()[2]
    
    def normalize_angle(self, angle):                                                     
        res = angle                                                                 
        while res > pi:                                                             
            res -= 2.0*pi                                                           
        while res < -pi:                                                            
            res += 2.0*pi                                                           
        return res
        
    def update_odom(self, msg):
        with self.lock:                                     
            #angle = self.quat_to_angle(msg.pose.pose.orientation)  
            q = msg.pose.pose.orientation
            self.q_odom_angle = PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)                                 
            #self.odom_time = msg.header.stamp  
        
    def shutdown(self):
        # Always stop the robot when shutting down the node
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
 
if __name__ == '__main__':
    try:
        CalibrateAngular()
    except rospy.ROSInterruptException:
        rospy.loginfo("Calibration terminated.")
