## ROS By Example - Electric to Fuerte PDF Conversion ##

**Please Note:** These instructions have only been tested under Ubuntu 11.10 (Oneric) and 12.04 (Precise)

  * Install the binary diff package `bsdiff` under Ubuntu:

> `$ sudo apt-get install bsdiff`

  * Move into the directory containing your ROS By Example PDF for Electric:

> `$ cd directory_containing_original_pdf`

There are three versions of the diff file, one for version 1.0 of the eBook and two for version 1.01 (depending on when you bought the book).  You can find the version at the bottom of the inside title page.

For Version 1.0 of the eBook:

  * Download the `electric2fuerte.diff` file:

> `$ wget http://ros-by-example.googlecode.com/files/electric2fuerte.diff`

For Version 1.01 of the eBook:

> `$ wget http://ros-by-example.googlecode.com/files/electric2fuerte_2.diff`

or

> `$ wget http://ros-by-example.googlecode.com/files/electric2fuerte_3.diff`

Try one and if that doesn't work, try the other.

  * Use the `bspatch` utility (part of the `bsdiff` package) to apply the diff to the Electric version of the PDF.  Change the source file name if you have renamed the original PDF and change the target file name if desired. (It can be anything you want but you should keep the .pdf extension.) For the 1.0 book version use:

> `$ bspatch ros_by_example___volume_1.pdf ros_by_example___volume_1_Fuerte.pdf electric2fuerte.diff`

and for the 1.01 version use:

> `$ bspatch ros_by_example___volume_1.pdf ros_by_example___volume_1_Fuerte.pdf electric2fuerte_2.diff`

or

> `$ bspatch ros_by_example___volume_1.pdf ros_by_example___volume_1_Fuerte.pdf electric2fuerte_3.diff`

> The general syntax is:

> `$ bspatch source_file output_file diff_file`

  * Bring up the new Fuerte-specific PDF in your favorite PDF reader.

  * Finally, don't forget to update your ROS By Example stack as some launch files have been added that are specific to Fuerte.  To update the stack, run the following commands:

```
      $ roscd rbx_vol_1
      $ svn update
      $ rosmake --pre-clean
```

**NOTE:** For future updates about the ROS By Example book and code, please join the [ros-by-example Google Group](https://groups.google.com/forum/#%21forum/ros-by-example)