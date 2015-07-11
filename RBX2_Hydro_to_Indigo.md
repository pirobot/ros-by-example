## ROS By Example Volume 2 Hydro to Indigo conversion ##

  * Install the binary diff package `bsdiff` under Ubuntu:

> `$ sudo apt-get install bsdiff`

  * Move into the directory containing your Volume 2 PDF for Hydro:

> `$ cd directory_containing_original_pdf`

  * Download the diff file appropriate for the version of your Hydro PDF.  You can find the version number on the inside title page and it should be either 1.01 or 1.02.  For version 1.01, use the command

> `$ wget -O rbx2-hydro-1.01-indigo.diff https://github.com/pirobot/ros-by-example/blob/wiki/diff/rbx2-hydro-1.01-indigo.diff?raw=true`

For version 1.02, use the command

> `$ wget -O rbx2-hydro-1.02-indigo.diff https://github.com/pirobot/ros-by-example/blob/wiki/diff/rbx2-hydro-1.02-indigo.diff?raw=true`

  * Use the `bspatch` utility (part of the `bsdiff` package) as shown below to apply the diff to your original PDF. Change the source file name (the first filename below) to match the filename of your original PDF and change the output file name (the second filename below) if desired. (It can be anything you want but you should keep the .pdf extension.)

For version 1.01 of the Hydro PDF, use:

> `$ bspatch ros_by_example_volume_2___hydro_1.01.pdf rbx2_vol_2_indigo.pdf rbx2-hydro-1.01-indigo.diff`

For version 1.02 of the Hydro PDF, use:

> `$ bspatch ros_by_example_volume_2___hydro_1.02.pdf rbx2_vol_2_indigo.pdf rbx2-hydro-1.02-indigo.diff`

> The general syntax is:

> `$ bspatch source_file output_file diff_file`

  * Bring up the new PDF in your favorite PDF reader.  The version number on the inner title page should be 1.1.0.

**NOTE:** For future updates about the ROS By Example books and code, please join the [ros-by-example Google Group](https://groups.google.com/forum/#%21forum/ros-by-example)
