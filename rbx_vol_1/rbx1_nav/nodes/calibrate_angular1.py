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

import roslib; roslib.load_manifest('rbx1_nav')
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import radians, copysign, pi
import PyKDL
import threading
from dynamic_reconfigure.server import Server
import dynamic_reconfigure.client
from rbx1_nav.cfg import CalibrateAngularConfig

class CalibrateAngular():
    def __init__(self):
        # Give the node a name
        rospy.init_node('calibrate_angular', anonymous=False)
        
        # Set rospy to exectute a shutdown function when terminating the script
        rospy.on_shutdown(self.shutdown)
        
        # Create a lock for reading odometry values
        self.lock = threading.Lock()
        
        # How fast will we check the odometry values?
        self.rate = 50
        r = rospy.Rate(self.rate)
        
        # The test angle is 360 degrees
        self.test_angle = radians(rospy.get_param('~test_angle', 360.0))

        self.speed = rospy.get_param('~speed', 0.7) # radians per second
        self.tolerance = rospy.get_param('tolerance', radians(5)) # degrees converted to radians
        self.odom_angular_scale_correction = rospy.get_param('~odom_angular_scale_correction', 1.0)
        self.target_reached = rospy.get_param('~target_reached', False)
        
        # Fire up the dynamic_reconfigure server
        dyn_server = Server(CalibrateAngularConfig, self.dynamic_reconfigure_callback)
        
        # Connect to the dynamic_reconfigure server
        dyn_client = dynamic_reconfigure.client.Client("calibrate_angular", timeout=60)
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist)
        
        # Subscribe to the odom topic and set the callback.  Remap the topic in the
        # launch file if using robot_pose_ekf
        rospy.Subscriber('odom', Odometry, self.update_odom)
        
        # Wait for the odom topic to become available
        rospy.wait_for_message('odom', Odometry)
        
        self.odom = Odometry()
        
        # Wait until we actually have some data
        while self.odom_angle is None:
            rospy.sleep(1)
            
        rospy.loginfo("Bring up dynamic_reconfigure to control the test.")
        
        reverse = 1
        
        while not rospy.is_shutdown():
            # Execute the rotation
            if not self.target_reached:
                last_angle = self.odom_angle
                turn_angle = 0
                
                # Alternate directions between tests
                reverse = -reverse
                angular_speed  = reverse * self.speed
                
                while abs(turn_angle) < abs(self.test_angle):
                    if rospy.is_shutdown():
                        return
                    move_cmd = Twist()
                    move_cmd.angular.z = angular_speed
                    self.cmd_vel.publish(move_cmd)
                    r.sleep()
                    
                    with self.lock:
                        delta_angle = self.odom_angular_scale_correction * normalize_angle(self.odom_angle - last_angle)
                    
                    turn_angle += delta_angle
                    last_angle = self.odom_angle
                
                # Stop the robot
                self.cmd_vel.publish(Twist())
                
                # Update the status flag
                self.target_reached = True
                params = {'target_reached': 'True'}
                dyn_client.update_configuration(params)
                
            rospy.sleep(0.5) 
                    
        # Stop the robot
        self.cmd_vel.publish(Twist())
        
    def update_odom(self, msg):
        with self.lock:                                     
            q = msg.pose.pose.orientation
            self.odom_angle = quat_to_angle(q)                                 
            
    def dynamic_reconfigure_callback(self, config, level):
        self.test_angle =  radians(config['test_angle'])
        self.speed = config['speed']
        self.tolerance = radians(config['tolerance'])
        self.odom_angular_scale_correction = config['odom_angular_scale_correction']
        self.target_reached = config['target_reached']
        
        return config
        
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
        CalibrateAngular()
    except:
        rospy.loginfo("Calibration terminated.")
