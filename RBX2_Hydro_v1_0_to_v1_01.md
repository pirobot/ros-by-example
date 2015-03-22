## ROS By Example Volume 2 Hydro: PDF version 1.0 to 1.01 ##

**Please Note:** These instructions have only been tested under Ubuntu 12.04 (Precise)

  * Install the binary diff package `bsdiff` under Ubuntu:

> `$ sudo apt-get install bsdiff`

  * Move into the directory containing your Volume 2 PDF for Hydro:

> `$ cd directory_containing_original_pdf`

  * Download the rbx2-hydro-1.0-to-1.01.diff file:

> `$ wget http://www.pirobot.org/ros/rbx/rbx2-hydro-1.0-to-1.01.diff`

  * Use the `bspatch` utility (part of the `bsdiff` package) as shown below to apply the diff to your original PDF. Change the source file name (the first filename below) to match the filename of your original PDF and change the target file name (the second filename below) if desired. (It can be anything you want but you should keep the .pdf extension.)

For example:

> `$ bspatch ros_by_example___volume_2.pdf rbx2_vol_2_hydro_v1.01.pdf rbx2-hydro-1.0-to-1.01.diff`

> The general syntax is:

> `$ bspatch source_file output_file diff_file`

  * Bring up the new PDF in your favorite PDF reader.  The version number on the inner title page should be 1.01.

**NOTE:** For future updates about the ROS By Example books and code, please join the [ros-by-example Google Group](https://groups.google.com/forum/#%21forum/ros-by-example)