#!/usr/bin/env python

"""
    follower.py - Version 1.0 2012-06-01
    
    Follow a "person" by tracking the nearest object in x-y-z space.
    
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
from sensor_msgs.msg import RegionOfInterest, CameraInfo, PointCloud2
from geometry_msgs.msg import Twist
from cv2 import cv as cv
import point_cloud2 as pc
from pointclouds import pointcloud2_to_array, get_xyz_points

class ObjectTracker():
    def __init__(self):
        rospy.init_node("person_follower")
        
        rospy.on_shutdown(self.shutdown)
        
        # What is our max rotation speed in radians per second?
        self.max_rotation_speed = rospy.get_param("~max_rotation_speed", 2.0)
        
        # What is our max linear speed in meters per second?
        self.max_linear_speed = rospy.get_param("~max_linear_speed", 1.0)
        
        # How quickly should we respond to angular target displacements?  Setting this too high
        # can lead to oscillations of the robot.
        self.k_pan = rospy.get_param("~k_angular", 1.5)
        
        # How quickly should we respond to linear target displacements?  Setting this too high
        # can lead to oscillations of the robot.
        self.k_pan = rospy.get_param("~k_linear", 0.5)
        
        # How often should we update our response to object motion?
        self.rate = rospy.get_param("~rate", 10)
        r = rospy.Rate(self.rate) 
        
        # The angular threshold indicates how far off-center the x-y ROI needs to be before we react
        self.angular_threshold = int(rospy.get_param("~angular_threshold", 0.1))
        
        # The linear threshold indicates how far away z ROI needs to be before we react
        self.angular_threshold = int(rospy.get_param("~linear_threshold", 0.1))

        # Publisher to control the robot's movement
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist)
        
        self.move_cmd = Twist()
        
        self.tracking_seq = 0
        self.last_tracking_seq = -1
        
        self.image_width = 0
        self.image_height = 0
        
        rospy.Subscriber('roi', RegionOfInterest, self.setCmdVel)
        rospy.Subscriber('camera_info', CameraInfo, self.getCameraInfo)
        rospy.Subscriber('point_cloud', PointCloud2, self.getPointCloud)
        
        # Wait for the camera_info topic to become available
        rospy.wait_for_message('camera_info', CameraInfo)

        # Wait until we actually have the camera data
        while self.image_width == 0 or self.image_height == 0:
            rospy.sleep(1)
        
        while not rospy.is_shutdown():
            # Update the robot's motion depending on the target's location
            if self.last_tracking_seq == self.tracking_seq:
                self.move_cmd = Twist()
            else:
                self.last_tracking_seq = self.tracking_seq
                
            self.cmd_vel_pub.publish(self.move_cmd)
            r.sleep()
    
    def setCmdVel(self, msg):
        # When OpenCV loses the ROI, the message stops updating.  Use this counter to
        # determine when it stops. 
        self.tracking_seq += 1

        # Compute the center of the ROI
        angular_offset = msg.x_offset + msg.width / 2 - self.image_width / 2
                  
        # Pan the camera only if the displacement of the COG exceeds the threshold
        if abs(angular_offset) > self.angular_threshold:
            # Set the rotation speed proportional to the displacement of the horizontal displacement
            # of the target
            try:
                self.move_cmd.angular.z = min(self.max_rotation_speed, self.k_angular * angular_offset / float(self.image_width))
            except:
                self.move_cmd = Twist()
        else:
            self.move_cmd = Twist()
            
    def getCameraInfo(self, msg):
        self.image_width = msg.width
        self.image_height = msg.height
        
    def getPointCloud(self, msg):
        x = y = z = n = 0
        t = cv.GetTickCount()
        
        point_array = pointcloud2_to_array(msg)
        points = get_xyz_points(point_array)
        
        for point in points:
            if abs(point[0]) < 0.3 and abs(point[1]) < 0.5 and abs(point[2]) < 1.5:
                x += point[0]
                y += point[1]
                z += point[2]
                n += 1
        
#        for point in pc.read_points(msg, skip_nans=True):
#            if abs(point[0]) < 0.3 and abs(point[1]) < 0.5:
#                x += point[0]
#                y += point[1]
#                z += point[2]
#                n += 1
            
        t = cv.GetTickCount() - t
        print "PC Time = %gms" % (t/(cv.GetTickFrequency()*1000.))
        
        if n > 0:    
            x /= n 
            y /= n 
            z /= n
            
        print msg.width * msg.height, x, y, z
        
    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)     
                   
if __name__ == '__main__':
    try:
        ObjectTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Object tracking node terminated.")




