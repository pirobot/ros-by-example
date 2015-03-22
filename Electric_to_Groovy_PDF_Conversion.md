## ROS By Example - Electric to Groovy PDF Conversion ##

**Please Note:** These instructions have only been tested under Ubuntu 11.10 (Oneric) and 12.04 (Precise)

  * Install the binary diff package `bsdiff` under Ubuntu:

> `$ sudo apt-get install bsdiff`

  * Move into the directory containing your ROS By Example PDF for Electric:

> `$ cd directory_containing_original_pdf`

There are **three** possible versions of the diff file, one for version 1.0 of the eBook and two for version 1.01.  Try first with the first diff file and if that does not work, try the second, then the third.

For Version 1.0 of the eBook:

  * Download the `electric2groovy.diff` file:

> `$ wget http://ros-by-example.googlecode.com/files/electric2groovy.diff`

For Version 1.01 of the eBook:

> `$ wget http://ros-by-example.googlecode.com/files/electric2groovy_2.diff`

or

> `$ wget http://ros-by-example.googlecode.com/files/electric2groovy_3.diff`

  * Use the `bspatch` utility (part of the `bsdiff` package) to apply the diff to the Electric version of the PDF.  Change the source file name if you have renamed the original PDF and change the target file name if desired. (It can be anything you want but you should keep the .pdf extension.)

For example, to try the first diff file use:

> `$ bspatch ros_by_example___volume_1.pdf ros_by_example___volume_1_Fuerte.pdf electric2groovy.diff`

If that does not work, try:

> `$ bspatch ros_by_example___volume_1.pdf ros_by_example___volume_1_Fuerte.pdf electric2groovy_2.diff`

And if that also doesn't work, try:

> `$ bspatch ros_by_example___volume_1.pdf ros_by_example___volume_1_Fuerte.pdf electric2groovy_3.diff`

> The general syntax is:

> `$ bspatch source_file output_file diff_file`

  * Bring up the new Groovy-specific PDF in your favorite PDF reader.

  * Finally, don't forget to get the new Groovy-specific ROS By Example sample code from Github as explained in Chapter 5:

**NOTE:** For future updates about the ROS By Example book and code, please join the [ros-by-example Google Group](https://groups.google.com/forum/#%21forum/ros-by-example)