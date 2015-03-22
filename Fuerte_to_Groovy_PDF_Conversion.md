## ROS By Example - Fuerte to Groovy PDF Conversion ##

**Please Note:** These instructions have only been tested under Ubuntu 12.04 (Precise)

  * Install the binary diff package `bsdiff` under Ubuntu:

> `$ sudo apt-get install bsdiff`

  * Move into the directory containing your ROS By Example PDF for Fuerte:

> `$ cd directory_containing_original_pdf`

**NOTE:**  There are **two** possible versions of the diff file depending on when you bought the Fuerte PDF.  Try first with the first diff file and if that does not work, try the second one.

To try the first diff file:

  * Download the `fuerte2groovy.diff` file:

> `$ wget http://ros-by-example.googlecode.com/files/fuerte2groovy.diff`

If this does not work when following the instructions below, then try the second diff file:

> `$ wget http://ros-by-example.googlecode.com/files/fuerte2groovy_2.diff`


  * Use the `bspatch` utility (part of the `bsdiff` package) to apply the diff to the Fuerte version of the PDF.  Change the source file name if you have renamed the original PDF and change the target file name if desired. (It can be anything you want but you should keep the .pdf extension.)

For example, to try the first diff file use:

> `$ bspatch ros_by_example___volume_1.pdf ros_by_example___volume_1_Groovy.pdf fuerte2groovy.diff`

If that does not work, try the second diff file:

> `$ bspatch ros_by_example___volume_1.pdf ros_by_example___volume_1_Groovy.pdf fuerte2groovy_2.diff`

> The general syntax is:

> `$ bspatch source_file output_file diff_file`

  * Bring up the new Groovy-specific PDF in your favorite PDF reader.

  * Finally, don't forget to get the new Groovy-specific ROS By Example sample code from Github as explained in Chapter 5:

**NOTE:** For future updates about the ROS By Example book and code, please join the [ros-by-example Google Group](https://groups.google.com/forum/#%21forum/ros-by-example)