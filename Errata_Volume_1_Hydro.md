# Errata for Volume 1 ROS Hydro #

  * There is a missing step in section **4.7 Mixing catkin and rosbuild   Workspaces**.

> In this section, you are instructed to add the following line to your ~/.bashrc file:

> `source ~/ros_workspace/setup.bash`

> The missing step is that we also need to add the following line right after the above line:

> `export ROS_PACKAGE_PATH=~/ros_workspace:$ROS_PACKAGE_PATH`

> This will ensure that your ~/ros\_workspace directory remains in your ROS\_PACKAGE\_PATH.

  * A bug has been fixed in the fake\_amcl.launch and fake\_nav\_test.launch files described in Sections 8.5.1 and 8.5.4.  Before the update, setting an initialpose for the fake TurtleBot in RViz did not work while running either fake\_amcl.launch or fake\_nav\_test.launch.  The bug has been fixed and you can get the updated files by running the commands:

```
$ roscd rbx1_nav
$ git pull
```

The updated text for Section 8.5.1 in the book should now read as follows in :

Finally, we fire up the fake\_localization node.  As you can read on the fake\_localization Wiki page, this node simulates a subset of the ROS amcl API in the case where localization is perfect.  However, we need to set three parameters to make it work with the ArbotiX simulator and our robot's URDF.  First we remap the base\_pose\_ground\_truth topic to the odom topic.  (Some robots use the odom\_combined topic if using more than one source of odometry data such as an IMU in addition to wheel encoders.)  When then set the parameter global\_frame\_id to the map frame and the base\_frame\_id parameter to base\_footprint.  If your robot does not use a base\_footprint frame, you would likely set this to base\_link instead.

  * At the end of Section 9.5, it is said that playing built-in sounds with sound\_play is not working in Hydro.  Apparently this has been fixed so built-in sounds now work again.  The updated paragraph in Section 9.5 now reads

To hear one of the built-in sounds, use the playbuiltin.py script together with a number from 1 to 5. Turn down your volume for this one!

```
$ rosrun sound_play playbuiltin.py 4
```

I'm not sure when the fix made it into the ROS Hydro Debian packages so you might have to update your Hydro packages if it's been awhile since you have done an update.  Don't forget that to hear the built-in sounds using the command above, you first have to have the main sound\_play node running:

```
$ rosrun sound_play soundplay_node.py
```

  * Referring to section 10.3.2, "Installing Webcam Drivers", instructions are provided for building Eric Perko's uvc\_cam package.  Eric's uvc\_cam package still uses rosbuild so for those of you who are or might have trouble running both rosbuild workspace and a catkin work spaces together, I have created a catkinized version of uvc\_cam.  Below are the steps for installing it in you your catkin workspace directory.

> NOTE: If you are already successfully using the rosbuild version of the uvc\_cam package, there is no need to install the catkinized version.  The actual driver files are identical.

> The following commands assume that your catkin workspace directory is located at ~/catkin\_ws.  Substitute the appropriate directory if your location is different.

```
  $ cd ~/catkin_ws/src
  $ git clone https://github.com/pirobot/uvc_cam.git
  $ cd uvc_cam
  $ git checkout hydro-devel
  $ cd ~/catkin_ws
  $ catkin_make
  $ rospack profile
```

> You can then proceed to test your webcam following the instructions in section 10.3.4.