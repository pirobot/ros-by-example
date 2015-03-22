The following errata are for book versions **1.0 - 1.02**.  They will be fixed in version 1.03.

  * In Section **6.4** page 181, the laptop\_battery.launch file is incorrectly described as including a rate parameter.  The rate is actually fixed in the laptop\_battery.py script so the rate parameter has been removed from the launch file and the text updated accordingly.

  * In Section **6.5** page 184, the roslaunch command has a typo (missing the "g" in "bringup").  It should be:

```
$ roslaunch rbx2_bringup laptop_battery.launch
```

  * In Section **11.8** (Installing MoveIt), instructions are provided for installing three MoveIt packages from source to get around a bug in moveit\_ros.  As of Feb 7 2015, that bug was fixed in the MoveIt Debian packages so that the source install is no longer necessary.  To install the Debian packages, run the following commands:

```
$ sudo apt-get update
$ sudo apt-get install ros-hydro-moveit-core ros-hydro-moveit-ros ros-hydro-moveit-setup-assistant
$ rospack profile
```

If you have already installed the source packages, you can remove them using the following commands:

```
$ cd ~/catkin_ws/src
$ \rm -rf moveit_core moveit_ros moveit_setup_assistant
$ cd ~/catkin_ws
$ catkin_make
$ rospack profile
```


The following errata are for book revisions **1.0** and **1.01**.  They have been **fixed in version 1.02**.  Check the inside title page for the revision number of your copy.

  * **Chapter** 2 _Installing the ROS By Example Code_

> There is a space missing in the long command for installing the prerequisite packages.  The missing space means copy-and-pasting the command to a terminal will fail.  The command should be:

```
$ sudo apt-get install ros-hydro-arbotix \
ros-hydro-dynamixel-motor ros-hydro-rosbridge-suite \
ros-hydro-mjpeg-server ros-hydro-rgbd-launch \
ros-hydro-openni-camera ros-hydro-moveit-full \
ros-hydro-turtlebot-* ros-hydro-kobuki-* ros-hydro-moveit-python \
python-pygraph python-pygraphviz python-easygui \
mini-httpd ros-hydro-laser-pipeline ros-hydro-ar-track-alvar \
ros-hydro-laser-filters ros-hydro-hokuyo-node \
ros-hydro-depthimage-to-laserscan ros-hydro-moveit-ikfast \
ros-hydro-gazebo-ros ros-hydro-gazebo-ros-pkgs \
ros-hydro-gazebo-msgs ros-hydro-gazebo-plugins \
ros-hydro-gazebo-ros-control ros-hydro-cmake-modules \
ros-hydro-kobuki-gazebo-plugins ros-hydro-kobuki-gazebo \
ros-hydro-smach ros-hydro-smach-ros ros-hydro-grasping-msgs \
ros-hydro-executive-smach ros-hydro-smach-viewer \
ros-hydro-robot-pose-publisher ros-hydro-tf2-web-republisher \
ros-hydro-move-base-msgs ros-hydro-fake-localization \
graphviz-dev libgraphviz-dev gv python-scipy sdformat
```

  * In Section **3.9.3** the blog posting by Bjoern Knafla mentioned at the end of section no longer exists.

  * In the **paperback** version of the book, the image on Page 257 is rather faint.  A better version of the image is shown below:

![http://www.pirobot.org/ros/rbx/ik.png](http://www.pirobot.org/ros/rbx/ik.png)

The rest of this errata is for **Version 1.0** of the book.  These issues have been fixed in version 1.01.

  * In Section **4.12.2** (_A fake Pi Robot_) in the first paragraph on page 145, there is a command missing from the text box. The missing command  is:

```
$ roslaunch rbx2_bringup pi_robot_with_gripper.launch sim:=true
```

A number of multi-line command examples are missing the continuation character (\) at the end of the first line.  The commands themselves are OK but copy-and-pasting these commands to a terminal will not work properly. Here are the relevant commands in a form that can be properly copy-and-pasted:

  * **Chapter 2 page 3**:

```
$ wget https://raw.githubusercontent.com/pirobot/rbx2/hydro-devel/rbx2-prereq.sh
```

  * **Section 4.12.1 page 148**:

```
$ rostopic pub -1 /right_arm_shoulder_lift_joint/command std_msgs/Float64 -- -1.57
```

```
$ rostopic pub -1 /right_arm_shoulder_lift_joint/command std_msgs/Float64 -- 0.0
```


  * **Section 5.4 page 165 and 166**:

```
$ rostopic pub -1 /right_arm_shoulder_lift_joint/command std_msgs/Float64 -- -2.0
```

```
$ rostopic pub -1 /right_arm_shoulder_lift_joint/command std_msgs/Float64 -- 0.0
```

  * **Section 5.5 page 167**:

```
$ roslaunch rbx2_bringup pi_robot_head_only.launch sim:=false port:=/dev/ttyUSB1
```

  * **Chapter 8 page 208**:

```
$ rosrun topic_tools mux cmd_vel move_base_cmd_vel joystick_cmd_vel mux:=mux_cmd_vel
```

  * **Section 8.2 page 211**:

```
$ rosrun topic_tools mux cmd_vel move_base_cmd_vel joystick_cmd_vel mux:=mux_cmd_vel
```

  * **Section 9.4.1 page 231**:

```
$ roslaunch rbx2_bringup pi_robot_head_only.launch sim:=false port:=/dev/ttyUSB0
```


  * **Section 11.16.2 page 308**:

```
$ rosrun rbx2_dynamixels head_trajectory_demo.py _reset:=true _duration:=10.0
```

  * **Section 11.25 page 354**:

```
$ rosrun rviz rviz -d `rospack find rbx2_arm_nav`/config/attached_object.rviz
```

  * **Section 11.26 page 357**:

```
$ rosrun rviz rviz -d `rospack find rbx2_arm_nav`/config/pick_and_place.rviz
```

  * **Section 11.29 page 378**:

```
$ rosrun moveit_ikfast round_collada_numbers.py pi_robot.dae pi_robot_rounded.dae 5
```

  * **Section 12.8 page 394**:

```
$ rostopic pub -r 10 /mobile_base/commands/velocity geometry_msgs/Twist  '{linear: {x: 0.2}}'
```

  * **Section 12.8 page 394**:

```
$ rostopic pub -r 10 /mobile_base/commands/velocity geometry_msgs/Twist '{linear: {x: 0.2}}'
```

  * **Section 12.8.1 page 395**:

```
$ rostopic pub -r 10 /mobile_base/commands/velocity geometry_msgs/Twist '{linear: {x: 0.1}}'
```

  * **Section 12.11.2 page 411**:

```
$ rosrun rviz rviz -d `rospack find rbx2_arm_nav`/config/real_pick_and_place.rviz
```


  * **Section 13.11 page 452**:

```
$ roslaunch rbx2_tasks fake_turtlebot.launch battery_runtime:=900 map:=test_map.yaml
```