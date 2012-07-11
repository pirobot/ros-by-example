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
from dynamic_reconfigure.server import Server
import dynamic_reconfigure.client
from rbx1_nav.cfg import CalibrateLinearConfig
import threading

class CalibrateLinear():
    def __init__(self):
        # Give the node a name
        rospy.init_node('calibrate_linear', anonymous=False)
        
        # Set rospy to exectute a shutdown function when terminating the script
        rospy.on_shutdown(self.shutdown)
        
        # Create a lock for reading odometry values
        self.lock = threading.Lock()
        
        # How fast will we check the odometry values?
        self.rate = 50
        r = rospy.Rate(self.rate)
        
        # Set the distance to travel
        self.test_distance = rospy.get_param('~test_distance', 1.0) # meters
        self.speed = rospy.get_param('~speed', 0.15) # meters per second
        self.tolerance = rospy.get_param('~tolerance', 0.01) # meters
        self.odom_linear_scale_correction = rospy.get_param('~odom_linear_scale_correction', 1.0)
        self.target_reached = rospy.get_param('~target_reached', False)
        
        # Fire up the dynamic_reconfigure server
        dyn_server = Server(CalibrateLinearConfig, self.dynamic_reconfigure_callback)
        
        # Connect to the dynamic_reconfigure server
        dyn_client = dynamic_reconfigure.client.Client("calibrate_linear", timeout=60)
        
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
            
        rospy.loginfo("Bring up dynamic_reconfigure to control the test.")
  
        odom_start = self.odom
        x_start = odom_start.pose.pose.position.x
        y_start = odom_start.pose.pose.position.y
            
        move_cmd = Twist()
            
        while not rospy.is_shutdown():
            # Stop the robot by default
            move_cmd = Twist()
            
            if not self.target_reached:
                # Compute the Euclidean distance from the target point
                with self.lock:
                    distance = sqrt(pow((x_start - self.odom.pose.pose.position.x), 2) +
                                    pow((y_start - self.odom.pose.pose.position.y), 2))
                
                # Correct the estimated distance by the correction factor
                distance *= self.odom_linear_scale_correction
                
                # How close are we?
                error =  distance - self.test_distance
                
                # Are we close enough?
                if self.target_reached or abs(error) <  self.tolerance:
                    self.target_reached = True
                    params = {'target_reached': 'True'}
                    dyn_client.update_configuration(params)
                else:
                    # If not, move in the appropriate direction
                    move_cmd.linear.x = copysign(self.speed, -1 * error)
            else:
                odom_start = self.odom
                x_start = odom_start.pose.pose.position.x
                y_start = odom_start.pose.pose.position.y
                
            self.cmd_vel.publish(move_cmd)
            r.sleep()

        # Stop the robot
        self.cmd_vel.publish(Twist())
        
    def dynamic_reconfigure_callback(self, config, level):
        self.test_distance = config['test_distance']
        self.speed = config['speed']
        self.tolerance = config['tolerance']
        self.odom_linear_scale_correction = config['odom_linear_scale_correction']
        self.target_reached = config['target_reached']
        
        return config
        
    def update_odom(self, msg):
        with self.lock:
            self.odom = msg
        
    def shutdown(self):
        # Always stop the robot when shutting down the node
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
 
if __name__ == '__main__':
    try:
        CalibrateLinear()
        rospy.spin()
    except:
        rospy.loginfo("Calibration terminated.")
