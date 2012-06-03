#!/usr/bin/env python

import numpy as np
import cv2
from common import anorm, clock
from functools import partial

help_message = '''SURF keypoints

USAGE: surf_keypoints.py [ <image> ]
'''

def display_keypoints(img1, p1): 
    green = (0, 255, 0)
    for (x1, y1) in np.int32(p1):
        cv2.circle(img1, (x1, y1), 3, green, -1)

if __name__ == '__main__':
    import sys
    try: fn1 = sys.argv[1]
    except:
        fn1 = "test_images/arch.jpg"
        
    print help_message

    img1 = cv2.imread(fn1, cv2.CV_LOAD_IMAGE_COLOR)

    start = clock()

    surf = cv2.SURF(600, _upright=True)
    kp1, desc1 = surf.detect(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None, False)

    displayed_kp1 = np.array([kp1[i].pt for i in range(len(kp1))])

    desc1.shape = (-1, surf.descriptorSize())
    print 'img1 - %d features' % (len(kp1))

    display_keypoints(img1, displayed_kp1)
    
    print "Elapsed time:", 1000 * (clock() - start)
    cv2.imshow('Display SURF Keypoints', img1)
    cv2.waitKey()
