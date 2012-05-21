#!/usr/bin/env python


""" match_template.py - Version 1.0 2012-02-28

    Find a template image within a test image
    
"""
import cv2.cv as cv
import cv2
import numpy as np
from time import clock

help_message = '''USAGE: pyrdown.py [<template_image>] [<test_image>] [<n_pyr>]

'''

if __name__ == '__main__':
    import sys
    try:
        target_file = sys.argv[1]
        test_file = sys.argv[2]
    except:
        target_file = "test_images/one_orl_face.jpg"
        test_file = "test_images/all_orl_faces_missing_one.jpg"
        
        #target_file = "test_images/melody_face_frontal.png"
        #test_file = "test_images/melody_test_8.png"
    
        print help_message
        
    try:
        n_pyr = int(sys.argv[3])
    except:
        n_pyr = 2
        
    # If we don't need different scales and orientations, set this to False
    scale_and_rotate = False
    show_all_matches = False
    ignore_threshold = True
        
    # Match threshold
    match_threshold = 0.7
        
    # Smallest template size in pixels we will consider
    min_template_size = 10
    max_template_size = 200
    
    # What multiplier should we use between adjacent scales
    scale_factor = 1.1 # 10% increases
    
    # Read in the template and test image
    template = cv2.imread(target_file, cv.CV_LOAD_IMAGE_COLOR)
    image = cv2.imread(test_file, cv.CV_LOAD_IMAGE_COLOR)
    
    #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if scale_and_rotate:
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
            if scale * max_template_dimension > max_template_size:
                break
        
        # And a set of rotation angles
        #rotations = [-60, -30, 0, 30, 60]
        rotations = [0]
    else:
        scales = [1]
        rotations = [0]

    # Add some noise to the image
#    rng = cv.RNG(-1)
#    noise = cv.CreateMat(image.shape[0], image.shape[1], cv.CV_8UC3)
#    cv.RandArr(rng, noise, noise.type, 200, 2)
#    noise_arr = np.array(noise, dtype=np.uint8)
#    image = cv2.add(image, noise_arr)

    # We need a copy of the original images for later work
    template_start = template.copy()
    image_copy = image.copy()
    
    # Make sure the template is smaller than the test image
    while template_start.shape[0] > image.shape[0] or template_start.shape[1] > image.shape[1]:
        template_start = cv2.resize(template_start, (int(0.5 * template_start.shape[0]), int(0.5 * template_start.shape[1])))
        
    # Use pyrDown() n_pyr times on the test image.  We only need to do this once for this image.
    for i in range(n_pyr):
        image_copy = cv2.pyrDown(image_copy)
    
    # Time how long this is going to take    
    start = clock()
    
    # Track which scale and rotation gives the best match
    maxScore = -1
    best_s = 1
    best_r = 0
    best_x = 0
    best_y = 0
    
    match_boxes = list()
    
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
            
            # Store the dimension of the transformed template and image
            h,w = template_copy.shape[:2]
            H,W = image_copy.shape[:2]
            
            result_width = W - w + 1
            result_height = H - h + 1
        
            # Run matchTemplate() on the reduced images
            result = cv2.matchTemplate(image_copy, template_copy, cv2.TM_CCOEFF_NORMED)
            
            # Squaring (or even cubing) the result exaggerates the differences
            display_result = np.abs(result)**2
            
            # Then do a sub-maximal suppression to emphasize the peaks
            #val, result = cv2.threshold(result, 0.01, 0, cv2.THRESH_TOZERO)
            
            # And normalize
            #result_normed = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # Find the maximum value on the result map
            (minValue, maxValue, minLoc, maxLoc) = cv2.minMaxLoc(result)
            
            if show_all_matches and maxValue > match_threshold:
                print "Match at: ", maxValue
                x, y = maxLoc
                x *= int(pow(2.0, n_pyr))
                y *= int(pow(2.0, n_pyr))
                h,w = template_start.shape[:2]
                h = int(h * s)
                w = int(w * s)
                match_boxes.append(((x + w/2, y + h/2), (w, h), -r))
                #cv2.imshow("Tmp Result" + str(r)+str(s), result)
                #cv2.imshow("Template" + str(s)+str(r), template_copy)

            if maxValue > maxScore:
                maxScore = maxValue
                best_x, best_y = maxLoc
                best_s = s
                best_r = r
                best_template = template_copy.copy()
                best_result = display_result.copy()
            
    # Transform back to original image sizes
    best_x *= int(pow(2.0, n_pyr))
    best_y *= int(pow(2.0, n_pyr))
    h,w = template_start.shape[:2]
    h = int(h * best_s)
    w = int(w * best_s)
    
    best_match_box = ((best_x + w/2, best_y + h/2), (w, h), -best_r)
    
    print "Best match found at scale:", best_s, "and rotation:", best_r, "and score: ",  maxScore
            
    # Stop the clock and print elapsed time
    elapsed = (clock() - start) * 1000
    print "Time elapsed: ", elapsed, "ms"

    best_result = cv2.resize(best_result, (int(pow(2.0, n_pyr)) * best_result.shape[1], int(pow(2.0, n_pyr)) * best_result.shape[0]))
    cv2.namedWindow("NCC", cv.CV_WINDOW_NORMAL)
    cv2.imshow("NCC", best_result)
    best_template = cv2.resize(best_template, (int(pow(2.0, n_pyr)) * best_template.shape[1], int(pow(2.0, n_pyr)) * best_template.shape[0]))
    cv2.imshow("Template", best_template)
#    cv2.imshow("Template", template_start)
#    cv2.imshow("Reduced Image", image_copy)
#    image_copy = cv2.resize(image_copy, (int(pow(2.0, n_pyr)) * image_copy.shape[1], int(pow(2.0, n_pyr)) * image_copy.shape[0]))
#    cv2.imshow("Reduced Image Magnified", image_copy)
    image_copy = cv2.resize(image_copy, (int(pow(2.0, n_pyr)) * image_copy.shape[1], int(pow(2.0, n_pyr)) * image_copy.shape[0]))
    # Draw a rotated ellipse around the best match location
    if show_all_matches and match_boxes:
        for match_box in match_boxes:
            cv2.ellipse(image, match_box, cv.RGB(255, 255, 50), 2)
        
    if maxScore > match_threshold or ignore_threshold:
        #cv2.ellipse(image, best_match_box, cv.RGB(50, 255, 50), 2)
        cv2.rectangle(image, (best_x, best_y), (best_x + w, best_y + h), cv.RGB(50, 255, 50), 4)
    
    cv2.imshow("Test Image", image)

    
#    cv2.imshow("Source Image", image)
#    cv2.imshow("PyrDown 3 Times", pyrdown_image)
#    cv2.imshow("Magnified PyrDown", display_image)
    
    cv2.waitKey()
    
    