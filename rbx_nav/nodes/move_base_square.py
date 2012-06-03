#!/usr/bin/python

""" move_base_square.py - Version 0.1 2012-01-10

    Command a robot to move in a square using move_base actions..

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
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker
from math import radians, pi

class MoveBaseSquare():
    def __init__(self):
        rospy.init_node('nav_test', anonymous=True)
        
        rospy.on_shutdown(self.shutdown)
        quaternions = list()
        
        # The first two target orientations are 90 degrees (horizontal pointing left)
        q_turn_angle = quaternion_from_euler(0, 0, pi / 2, axes='sxyz')
        q = Quaternion()
        q.x = q_turn_angle[0]
        q.y = q_turn_angle[1]
        q.z = q_turn_angle[2]
        q.w = q_turn_angle[3]
        quaternions.append(q)
        quaternions.append(q)

        # The second two target orientations are 270 degrees (horizontal point right)
        q_turn_angle = quaternion_from_euler(0, 0, 3 * pi / 2, axes='sxyz')
        q = Quaternion()
        q.x = q_turn_angle[0]
        q.y = q_turn_angle[1]
        q.z = q_turn_angle[2]
        q.w = q_turn_angle[3]
        quaternions.append(q)
        quaternions.append(q)

        
        waypoints = list()
        waypoints.append(Pose(Point(1.0, 0.0, 0.0), quaternions[0]))
        waypoints.append(Pose(Point(1.0, 1.0, 0.0), quaternions[1]))
        waypoints.append(Pose(Point(0.0, 1.0, 0.0), quaternions[2]))
        waypoints.append(Pose(Point(0.0, 0.0, 0.0), quaternions[3]))
        
        # Initialize the waypoint markers
        self.init_markers()
        
        # Set a marker at each location        
        for waypoint in waypoints:           
            p = Point()
            p = waypoint.position
            self.markers.points.append(p)
            
        # Publisher to manually control the robot (e.g. to stop it)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist)
        
        # Subscribe to the move_base action server
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Connected to move base server")
            
        rospy.loginfo("Starting navigation test")
        
        for i in range(4):
            self.marker_pub.publish(self.markers)
            self.goal = MoveBaseGoal()
            self.goal.target_pose.pose = waypoints[i]
            self.goal.target_pose.header.frame_id = 'map'
            self.goal.target_pose.header.stamp = rospy.Time.now()
            self.move_base()
        
    def move_base(self):
            # Start the robot toward the next location
            self.move_base.send_goal(self.goal)
            
            # Allow 1 minute to get there
            finished_within_time = self.move_base.wait_for_result(rospy.Duration(60)) 
            
            if not finished_within_time:
                self.move_base.cancel_goal()
                rospy.loginfo("Timed out achieving goal")
            else:
                state = self.move_base.get_state()
                if state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("Goal succeeded!")
                    
    def init_markers(self):
        # Set up our waypoint markers
        marker_scale = 0.1
        marker_lifetime = 0 # 0 is forever
        marker_ns = 'waypoints'
        marker_id = 0
        marker_color = {'r': 1.0, 'g': 0.0, 'b': 0.0, 'a': 1.0}
        
        # Define a marker publisher.
        self.marker_pub = rospy.Publisher('waypoint_markers', Marker)
        
        # Initialize the marker points list.
        self.markers = Marker()
        self.markers.ns = marker_ns
        self.markers.id = marker_id
        self.markers.type = Marker.CUBE_LIST
        self.markers.action = Marker.ADD
        self.markers.lifetime = rospy.Duration(marker_lifetime)
        self.markers.scale.x = marker_scale
        self.markers.scale.y = marker_scale
        self.markers.color.r = marker_color['r']
        self.markers.color.g = marker_color['g']
        self.markers.color.b = marker_color['b']
        self.markers.color.a = marker_color['a']
        
        self.markers.header.frame_id = 'map'
        self.markers.header.stamp = rospy.Time.now()
        self.markers.points = list()

    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        self.move_base.cancel_goal()
        rospy.sleep(2)
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)

if __name__ == '__main__':
    try:
        MoveBaseSquare()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
