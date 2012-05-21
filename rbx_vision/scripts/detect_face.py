#!/usr/bin/python
"""
This program is demonstration for face and object detection using haar-like features.
The program finds faces in a camera image or video stream and displays a red box around them.

Original C implementation by:  ?
Python implementation by: Roman Stanchak, James Bowman
"""
import sys
from cv2 import cv as cv
import cv2
from optparse import OptionParser

# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned 
# for accurate yet slow object detection. For a faster operation on real video 
# images the settings are: 
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING, 
# min_size=<minimum possible face size

min_size = (20, 20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0

def detect_and_draw(img):
    # allocate temporary images
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),
			       cv.Round (img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
    
    cv.Smooth(gray, gray)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    # Equalize to minimize lighting effects
    cv.EqualizeHist(small_img, small_img)

    if(cascade1):
        t = cv.GetTickCount()
        faces = cv.HaarDetectObjects(small_img, cascade1, cv.CreateMemStorage(0),
                                     haar_scale, min_neighbors, haar_flags, min_size)
        if not faces and cascade2:
            faces = cv.HaarDetectObjects(small_img, cascade2, cv.CreateMemStorage(0),
                                     haar_scale, min_neighbors, haar_flags, min_size)
            
        t = cv.GetTickCount() - t
        print "detection time = %gms" % (t/(cv.GetTickFrequency()*1000.))
        if faces:
            for ((x, y, w, h), n) in faces:
                # the input to cv.HaarDetectObjects was resized, so scale the 
                # bounding box of each face and convert it to two CvPoints
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                cv.Rectangle(img, pt1, pt2, cv.RGB(50, 255, 50), 3, 8, 0)

    cv.ShowImage("Faces", img)

if __name__ == '__main__':
    import sys
    try: image_file = sys.argv[1]
    except:
        image_file = "test_images/cat.jpg"
        
    image = cv.LoadImage(image_file, 1)

    cascade1= cv.Load("../models/haar_face_detector/haarcascade_frontalface_alt.xml")
    cascade2= cv.Load("../models/haar_face_detector/haarcascade_profileface.xml")

    cv.NamedWindow("Faces", cv.CV_WINDOW_NORMAL)
    cv.ResizeWindow("Faces", image.width, image.height)

    detect_and_draw(image)
    cv.WaitKey(0)

    cv.DestroyWindow("Faces")
    
