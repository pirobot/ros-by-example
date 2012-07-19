#!/usr/bin/env python

"""
    Relax all servos by disabling the torque for each.
"""
import roslib
roslib.load_manifest('rbx1_dynamixels')
import rospy, time
from dynamixel_controllers.srv import TorqueEnable, SetSpeed

class Relax():
    def __init__(self):
        rospy.init_node('relax_all_servos')
        
        dynamixels = rospy.get_param('dynamixels', '')
        
        torque_services = list()
        speed_services = list()
            
        for name in sorted(dynamixels):
            controller = name.replace("_joint", "") + "_controller"
            
            torque_service = '/' + controller + '/torque_enable'
            rospy.wait_for_service(torque_service)  
            torque_services.append(rospy.ServiceProxy(torque_service, TorqueEnable))
            
            speed_service = '/' + controller + '/set_speed'
            rospy.wait_for_service(speed_service)  
            speed_services.append(rospy.ServiceProxy(speed_service, SetSpeed))
        
        # Set the default speed to something small
        for set_speed in speed_services:
            try:
                set_speed(0.1)
            except:
                pass

        # Relax all servos to give them a rest.
        for torque_enable in torque_services:
            try:
                torque_enable(False)
            except:
                pass
        
if __name__=='__main__':
    Relax()
