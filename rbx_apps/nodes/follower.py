#!/usr/bin/env python

"""
    follower.py - Version 1.0 2012-06-01
    
    Follow a "person" by tracking the nearest object in x-y-z space.
    
    Based on the follower application by Tony Pratkanis at:
    
    http://ros.org/wiki/turtlebot_follower
    
    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2012 Patrick Goebel.  All rights reserved.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
"""

import roslib; roslib.load_manifest('rbx_apps')
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist, Point
from cv2 import cv as cv
import point_cloud2 as pc
from math import copysign
#from pointclouds import pointcloud2_to_array, get_xyz_points

class Follower():
    def __init__(self):
        rospy.init_node("follower")
        
        # Set the shutdown function (stop the robot)
        rospy.on_shutdown(self.shutdown)
        
        # The min/max dimensions (in meters) of the person (blob).  These are given in camera coordinates
        # where x is left/right, y is up/down and z is depth (forward/backward)
        self.min_x = rospy.get_param("~min_x", -0.2)
        self.max_x = rospy.get_param("~max_x", 0.2)
        self.min_y = rospy.get_param("~min_y", 0.1)
        self.max_y = rospy.get_param("~max_y", 0.5)
        self.max_z = rospy.get_param("~max_z", 0.8)
        
        # The goal distance (in meters) to keep between the robot and the person
        self.goal_z = rospy.get_param("~goal_z", 0.6)
        
        # How far away from the goal distance before the robot reacts
        self.z_threshold = rospy.get_param("~z_threshold", 0.05)
        
        # How far away being centered (x displacement) on the person before the robot reacts
        self.x_threshold = rospy.get_param("~x_threshold", 0.1)
        
        # How much to weight the goal distance (z) when making a movement
        self.z_scale = rospy.get_param("~z_scale", 1.0)

        # How much to weight x-displacement of the person when making a movement        
        self.x_scale = rospy.get_param("~x_scale", 5.0)
        
        # The maximum rotation speed in radians per second
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 2.0)
        
        # The minimum rotation speed in radians per second
        self.min_angular_speed = rospy.get_param("~min_angular_speed", 0.5)
        
        # The max linear speed in meters per second
        self.max_linear_speed = rospy.get_param("~max_linear_speed", 0.5)
        
        # The minimum linear speed in meters per second
        self.min_linear_speed = rospy.get_param("~min_linear_speed", 0.1)
    
        # Publisher to control the robot's movement
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist)

        rospy.Subscriber('/camera/depth/points', PointCloud2, self.getPointCloud)
        
        # Wait for the pointcloud topic to become available
        rospy.wait_for_message('/camera/depth/points', PointCloud2)
    
    def setCmdVel(self, msg):
        # Compute the center of the ROI
        angular_offset = msg.x_offset + msg.width / 2 - self.image_width / 2
                  
        # Pan the camera only if the displacement of the COG exceeds the threshold
        if abs(angular_offset) > self.angular_threshold:
            # Set the rotation speed proportional to the displacement of the horizontal displacement
            # of the target
            try:
                self.move_cmd.angular.z = min(self.max_angular_speed, self.k_angular * angular_offset / float(self.image_width))
            except:
                self.move_cmd = Twist()
        else:
            self.move_cmd = Twist()
        
    def getPointCloud(self, msg):
        x = y = z = n = 0
        
        #t = cv.GetTickCount()
        
#        point_array = pointcloud2_to_array(msg)
#        points = get_xyz_points(point_array)
#        
#        for point in points:
#            if abs(point[0]) < 0.3 and abs(point[1]) < 0.5 and abs(point[2]) < 1.5:
#                x += point[0]
#                y += point[1]
#                z += point[2]
#                n += 1
        
        for point in pc.read_points(msg, skip_nans=True):
            pt_x = point[0]
            pt_y = point[1]
            pt_z = point[2]
            
            if -pt_y > self.min_y and -pt_y < self.max_y and  pt_x < self.max_x and pt_x > self.min_x and pt_z < self.max_z:
                x += pt_x
                y += pt_y
                z += pt_z
                n += 1
            
        #t = cv.GetTickCount() - t
        #print "PC Time = %gms" % (t/(cv.GetTickFrequency()*1000.))
        
        # Stop the robot by default
        move_cmd = Twist()
        
        if n:    
            x /= n 
            y /= n 
            z /= n
            
            rospy.loginfo("Centriod at %f %f %f with %d points", x, y, z, n)
      
            # Compute the linear and angular components of the movement
            linear_speed = (z - self.goal_z) * self.z_scale
            angular_speed = -x * self.x_scale
            
            # Make sure we meet our min/max specifications
            linear_speed = copysign(max(self.min_linear_speed, min(self.max_linear_speed, abs(linear_speed))), linear_speed)
            angular_speed = copysign(max(self.min_angular_speed, min(self.max_angular_speed, abs(angular_speed))), angular_speed)

            move_cmd.linear.x = linear_speed
            move_cmd.angular.z = angular_speed
            
        # Publish the movement command
        self.cmd_vel_pub.publish(move_cmd)

        
    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)     
                   
if __name__ == '__main__':
    try:
        Follower()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Follower node terminated.")




