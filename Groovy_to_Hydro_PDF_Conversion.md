## ROS By Example - Groovy to Hydro PDF Conversion ##

**Please Note:** These instructions have only been tested under Ubuntu 12.04 (Precise)

  * Install the binary diff package `bsdiff` under Ubuntu:

> `$ sudo apt-get install bsdiff`

  * Move into the directory containing your ROS By Example PDF for Groovy:

> `$ cd directory_containing_original_pdf`

  * Download the groovy2hydro.diff file:

> `$ wget http://ros-by-example.googlecode.com/files/groovy2hydro.diff`

  * Use the `bspatch` utility (part of the `bsdiff` package) to apply the diff to the Grpovy version of the PDF.  Change the source file name if you have renamed the original PDF and change the target file name if desired. (It can be anything you want but you should keep the .pdf extension.)

For example:

> `$ bspatch ros_by_example___volume_1.pdf ros_by_example___volume_1_hydro.pdf groovy2hydro.diff`

> The general syntax is:

> `$ bspatch source_file output_file diff_file`

  * Bring up the new Hydro-specific PDF in your favorite PDF reader.

  * Finally, don't forget to get the new Hydro-specific ROS By Example sample code from Github as explained in Chapter 5:

**NOTE:** For future updates about the ROS By Example book and code, please join the [ros-by-example Google Group](https://groups.google.com/forum/#%21forum/ros-by-example)