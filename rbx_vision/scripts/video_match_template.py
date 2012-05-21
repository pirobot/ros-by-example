#!/usr/bin/env python


""" video_match_template.py - Version 1.0 2012-02-28

    Find a template image within a video stream
    
"""
import cv2.cv as cv
import cv2
import numpy as np
from time import clock
import video

help_message = '''USAGE: pyrdown.py [<template_image>] [video_src>] [<n_pyr>]

'''

if __name__ == '__main__':
    import sys
    try:
        template_fn = sys.argv[1]
        video_src = sys.argv[2]
    except:
        template_fn = "test_images/mona_lisa_face.png"
        image_fn = "test_images/mona_lisa.png"
        print help_message
        
    try:
        n_pyr = int(sys.argv[3])
    except:
        n_pyr = 3
        
    capture = video.create_capture(video_src)
        
    # If we don't need different scales and orientations, set this to False
    use_variations = False
        
    # Match threshold
    match_threshold = 0.8
        
    # Smallest template size in pixels we will consider
    min_template_size = 25 
    
    # What multiplier should we use between adjacent scales
    scale_factor = 1.2 # 20% increases
    
    # Read in the template and test image
    template = cv2.imread(template_fn, cv.CV_LOAD_IMAGE_COLOR)
    
    # Get an intial frame
    ret, image = capture.read()
    
    if use_variations:
        # Compute the min and max scales to use on the template
        height_ratio = float(image.shape[0]) / template.shape[0]
        width_ratio = float(image.shape[1]) / template.shape[1]
        
        max_scale = 0.9 * min(width_ratio, height_ratio)
        
        max_template_dimension = max(template.shape[0], template.shape[1])
        min_scale = 1.1 * float(min_template_size) / max_template_dimension
        
        # Create a list of scales we will use
        scales = list()
        scale = min_scale
        while scale < max_scale:
            scales.append(scale)
            scale *= scale_factor
        
        # And a set of rotation angles
        rotations = [-45, 0, 45]
    else:
        scales = [1]
        rotations = [0]
        
    # Add some noise to the image
#    rng = cv.RNG(-1)
#    noise = cv.CreateMat(image.shape[0], image.shape[1], cv.CV_8UC3)
#    cv.RandArr(rng, noise, noise.type, 200, 2)
#    noise_arr = np.array(noise, dtype=np.uint8)
#    image = cv2.add(image, noise_arr)

    cv2.namedWindow("Image", cv.CV_WINDOW_NORMAL)

    ret, image = capture.read()

    # We need a copy of the template image for later work
    template_start = template.copy()
    
    # Make sure the template is smaller than the test image
    while template_start.shape[0] > image.shape[0] or template_start.shape[1] > image.shape[1]:
        template_start = cv2.resize(template_start, (int(0.5 * template_start.shape[0]), int(0.5 * template_start.shape[1])))
    
    while True:
        ret, image = capture.read()
        image_copy = image.copy()

        # Time how long this is going to take    
        start = clock()
        
        # Track which scale and rotation gives the best match
        maxScore = -1
        best_s = 1
        best_r = 0
        best_x = 0
        best_y = 0
        
        for s in scales:
            for r in rotations:
                template_height, template_width  = template_start.shape[0], template_start.shape[1]
                
                # Scale the template by s
                template_copy = cv2.resize(template_start, (int(template_width * s), int(template_height * s)))
                
                # Rotate the template through r degrees
                rotation_matrix = cv2.getRotationMatrix2D((template_copy.shape[1]/2, template_copy.shape[0]/2), r, 1.0)
                template_copy = cv2.warpAffine(template_copy, rotation_matrix, (template_copy.shape[1], template_copy.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    
                # Use pyrDown() n_pyr times on the scaled and rotated template
                for i in range(n_pyr):
                    template_copy = cv2.pyrDown(template_copy)
                
                # Create the results array to be used with matchTempate()
                h,w = template_copy.shape[:2]
                H,W = image_copy.shape[:2]
                
                result_width = W - w + 1
                result_height = H - h + 1
                
                result_mat = cv.CreateMat(result_height, result_width, cv.CV_32FC1)
                result = np.array(result_mat, dtype = np.float32)
            
                # Run matchTemplate() on the reduced images
                cv2.matchTemplate(image_copy, template_copy, cv.CV_TM_CCOEFF_NORMED, result)
                
                # Find the maximum value on the result map
                (minValue, maxValue, minLoc, maxLoc) = cv2.minMaxLoc(result)
                
                if maxValue > maxScore:
                    maxScore = maxValue
                    best_x, best_y = maxLoc
                    best_s = s
                    best_r = r
                    best_template = template_copy.copy()
                
        # Transform back to original image sizes
        best_x *= int(pow(2.0, n_pyr))
        best_y *= int(pow(2.0, n_pyr))
        h,w = template_start.shape[:2]
        h = int(h * best_s)
        w = int(w * best_s)
        
        match_box = ((best_x + w/2, best_y + h/2), (w, h), -best_r)
        
        print "Best match found at scale:", best_s, "and rotation:", best_r, "and score: ",  maxScore
        print best_x, best_y
        
        # Draw a rectangle around the best match location             
        cv2.rectangle(image, (best_x, best_y), (best_x + w, best_y + h), cv.RGB(50, 255, 50), 3)
        
        cv2.imshow("Image", image)
        
        # Draw a rotated ellipse around the best match location
        #cv.EllipseBox(cv.fromarray(image), match_box, cv.RGB(50, 255, 50), 3)
        #cv2.ellipse(image, match_box, cv.RGB(50, 255, 50), 3)
            
        # Stop the clock and print elapsed time
        elapsed = (clock() - start) * 1000
        #print "Time elapsed: ", elapsed, "ms"
        

        
        cv2.waitKey(5)


#    result = cv2.resize(result, (int(pow(2.0, n_pyr)) * result.shape[1], int(pow(2.0, n_pyr)) * result.shape[0]))
#    cv2.imshow("Result", result)
#    cv2.imshow("Reduced Template", template_copy)
#    best_template = cv2.resize(best_template, (int(pow(2.0, n_pyr)) * best_template.shape[1], int(pow(2.0, n_pyr)) * best_template.shape[0]))
#    cv2.imshow("Reduced Template Magnified", best_template)
#    cv2.imshow("Template", template_start)
#    cv2.imshow("Reduced Image", image_copy)
#    image_copy = cv2.resize(image_copy, (int(pow(2.0, n_pyr)) * image_copy.shape[1], int(pow(2.0, n_pyr)) * image_copy.shape[0]))
#    cv2.imshow("Reduced Image Magnified", image_copy)
#    cv2.imshow("Image", image)

    
#    cv2.imshow("Source Image", image)
#    cv2.imshow("PyrDown 3 Times", pyrdown_image)
#    cv2.imshow("Magnified PyrDown", display_image)
    
#    cv2.waitKey()
    
    