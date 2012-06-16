#!/usr/bin/env python

import numpy as np
import cv2
from cv2 import cv as cv
from common import anorm, clock
from functools import partial
import random
from math import sin, cos

help_message = '''SURF image match 

USAGE: surf_find_obj.py [ <image1> <image2> ]
'''

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing

flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 4)

n_warps = 10
noise_var = 1

def warp_target(img):
    samples = list()
    warp = cv.CreateMat(2, 3, cv.CV_32FC1)    
    ran = random.Random()
    sample = cv.CloneMat(cv.fromarray(img))
    
    noise = cv.CloneMat(sample)
    #grey = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)
    
    cv.Zero(noise)
    cv.RandArr(cv.RNG(), noise, cv.CV_RAND_NORMAL, (0, 0, 0), (noise_var, noise_var, noise_var))

    cv.Add(sample, noise, sample)
    #cv.CvtColor(sample, grey, cv.CV_RGB2GRAY)
    
    #cv.EqualizeHist(grey, grey)
    cv.EqualizeHist(sample, sample)

#        cv.FloodFill(grey, (grey.width/2, grey.height/2), 200, lo_diff=(0, 0, 0, 0), up_diff=(100, 100, 100, 100), flags=4, mask=None)
#        edges = cv.CloneMat(grey)
#        cv.Canny(grey, edges, 0, 200)
#        cv.Dilate(grey, grey, None, 1)
#        
#        contour = cv.FindContours(edges, cv.CreateMemStorage(0), mode=cv.CV_RETR_EXTERNAL)
#        
#        cv.DrawContours(edges, contour, cv.RGB(255, 255, 255), cv.RGB(255, 255, 255), 1, 5)
#        cv.ShowImage("Edges", edges)

    result = cv.CloneMat(sample)

    mask = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)
    warped_mask = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)
    cv.Set(mask, cv.ScalarAll(255))
    
    # TODO: Is this necessary? Remove border pixels from mask
    for i in range(mask.cols):
        cv.Set2D(mask, 0, i, 0)
        cv.Set2D(mask, mask.rows - 1, i, 0)
    for j in range(mask.rows):
        cv.Set2D(mask, j, 0, 0)
        cv.Set2D(mask, j, mask.cols - 1, 0)
    
    param_theta = 0.2
    param_sin_cos = 0.2
    for i in range(n_warps):
        theta = ran.uniform(-param_theta, param_theta)
        sx = cos(theta) + ran.uniform(-param_sin_cos, param_sin_cos)
        rx = -sin(theta) + ran.uniform(-param_sin_cos, param_sin_cos)
        sy = sin(theta) + ran.uniform(-param_sin_cos, param_sin_cos)
        ry = cos(theta) + ran.uniform(-param_sin_cos, param_sin_cos)
        tx = (1 - sx) * sample.width / 2 - rx * sample.height / 2
        ty = (1 - sy) * sample.height / 2 - ry * sample.width / 2
        cv.Set2D(warp, 0, 0, sx)  
        cv.Set2D(warp, 0, 1, rx)
        cv.Set2D(warp, 0, 2, tx)
        cv.Set2D(warp, 1, 0, sy)  
        cv.Set2D(warp, 1, 1, ry)
        cv.Set2D(warp, 1, 2, ty)

        # TODO: replace WarpAffine with WarpPersective
        cv.WarpAffine(sample, result, warp, fillval=0)
#        cv.WarpAffine(mask, warped_mask, warp)
                    
#            eig = cv.CreateImage(cv.GetSize(grey), 32, 1)
#            temp = cv.CreateImage(cv.GetSize(grey), 32, 1)
#            
#            features = []
#            
#            features = cv.GoodFeaturesToTrack(result, eig, temp, self.max_count,
#                    self.quality, self.good_feature_distance, mask=None, blockSize=3, useHarris=0, k=0.04)

#        result_array = np.array(result, dtype=np.uint8)
#        mask_array = np.array(warped_mask, dtype=np.uint8)
#        keypoints, descriptors =  surf.detect(result_array, mask_array, False)
#        
#        displayed_keypoints = np.array([keypoints[i].pt for i in range(len(keypoints))])
#        for (x1, y1) in np.int32(displayed_keypoints):
#            cv.Circle(result, (int(x1), int(y1)), 2, (255, 255, 255, 0), cv.CV_FILLED, 8, 0)
#       
#        cv.ShowImage("Warped", result)                  
#        cv.WaitKey(0)
        
        samples.append(result)
      
    return samples

#def generate_samples(kp, desc):
#    warp = cv.CreateMat(2, 3, cv.CV_32FC1)    
#    ran = random.Random()
#    sample = cv.CloneMat(cv.fromarray(img1))
#    
#    noise = cv.CloneMat(sample)
#    grey = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)
#    
#    cv.Zero(noise)
#    cv.RandArr(cv.RNG(), noise, cv.CV_RAND_NORMAL, (0, 0, 0), (noise_var, noise_var, noise_var))
#
#    cv.Add(sample, noise, sample)
#    cv.CvtColor(sample, grey, cv.CV_RGB2GRAY)
#    
#    cv.EqualizeHist(grey, grey)
#
##        cv.FloodFill(grey, (grey.width/2, grey.height/2), 200, lo_diff=(0, 0, 0, 0), up_diff=(100, 100, 100, 100), flags=4, mask=None)
##        edges = cv.CloneMat(grey)
##        cv.Canny(grey, edges, 0, 200)
##        cv.Dilate(grey, grey, None, 1)
##        
##        contour = cv.FindContours(edges, cv.CreateMemStorage(0), mode=cv.CV_RETR_EXTERNAL)
##        
##        cv.DrawContours(edges, contour, cv.RGB(255, 255, 255), cv.RGB(255, 255, 255), 1, 5)
##        cv.ShowImage("Edges", edges)
#
#    result = cv.CloneMat(grey)
#
#    mask = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)
#    warped_mask = cv.CreateMat(sample.rows, sample.cols, cv.CV_8U)
#    cv.Set(mask, cv.ScalarAll(255))
#    
#    # TODO: Is this necessary? Remove border pixels from mask
#    for i in range(mask.cols):
#        cv.Set2D(mask, 0, i, 0)
#        cv.Set2D(mask, mask.rows - 1, i, 0)
#    for j in range(mask.rows):
#        cv.Set2D(mask, j, 0, 0)
#        cv.Set2D(mask, j, mask.cols - 1, 0)
#    
#    param_theta = 0.2
#    param_sin_cos = 0.2
#    for i in range(n_warps):
#        theta = ran.uniform(-param_theta, param_theta)
#        sx = cos(theta) + ran.uniform(-param_sin_cos, param_sin_cos)
#        rx = -sin(theta) + ran.uniform(-param_sin_cos, param_sin_cos)
#        sy = sin(theta) + ran.uniform(-param_sin_cos, param_sin_cos)
#        ry = cos(theta) + ran.uniform(-param_sin_cos, param_sin_cos)
#        tx = (1 - sx) * sample.width / 2 - rx * sample.height / 2
#        ty = (1 - sy) * sample.height / 2 - ry * sample.width / 2
#        cv.Set2D(warp, 0, 0, sx)  
#        cv.Set2D(warp, 0, 1, rx)
#        cv.Set2D(warp, 0, 2, tx)
#        cv.Set2D(warp, 1, 0, sy)  
#        cv.Set2D(warp, 1, 1, ry)
#        cv.Set2D(warp, 1, 2, ty)
#
#        # TODO: replace WarpAffine with WarpPersective
#        cv.WarpAffine(grey, result, warp, fillval=0)
#        cv.WarpAffine(mask, warped_mask, warp)
#                    
##            eig = cv.CreateImage(cv.GetSize(grey), 32, 1)
##            temp = cv.CreateImage(cv.GetSize(grey), 32, 1)
##            
##            features = []
##            
##            features = cv.GoodFeaturesToTrack(result, eig, temp, self.max_count,
##                    self.quality, self.good_feature_distance, mask=None, blockSize=3, useHarris=0, k=0.04)
#
#        result_array = np.array(result, dtype=np.uint8)
#        mask_array = np.array(warped_mask, dtype=np.uint8)
#        keypoints, descriptors =  surf.detect(result_array, mask_array, False)
#        #print(len(keypoints))
#        
#        kp = np.concatenate((kp, keypoints))
#        desc = np.concatenate((desc, descriptors))
#        
#        displayed_keypoints = np.array([keypoints[i].pt for i in range(len(keypoints))])
#        for (x1, y1) in np.int32(displayed_keypoints):
#            cv.Circle(result, (int(x1), int(y1)), 2, (255, 255, 255, 0), cv.CV_FILLED, 8, 0)
#       
#        cv.ShowImage("Warped", result)
#        cv.WaitKey(0)
#
#        
#    return kp, desc

def match_bruteforce(desc1, desc2, r_threshold = 0.75):
    res = []
    for i in xrange(len(desc1)):
        dist = anorm( desc2 - desc1[i] )
        n1, n2 = dist.argsort()[:2]
        r = dist[n1] / dist[n2]
        if r < r_threshold:
            res.append((i, n1))
    return np.array(res)

def match_flann(desc1, desc2, r_threshold = 0.6):
    flann = cv2.flann_Index(desc2, flann_params)
    idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
    mask = dist[:,0] / dist[:,1] < r_threshold
    idx1 = np.arange(len(desc1))
    pairs = np.int32( zip(idx1, idx2[:,0]) )
    return pairs[mask]

def draw_match(img1, img2, p1, p2, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    #vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (50, 255, 50), 3)
    
    if status is None:
        status = np.ones(len(p1), np.bool_)
    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(np.int32(p1), np.int32(p2), status):
        col = [red, green][inlier]
        if inlier:
            cv2.line(vis, (x1, y1), (x2+w1, y2), col)
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2+w1, y2), 2, col, -1)
        else:
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2+w1-r, y2-r), (x2+w1+r, y2+r), col, thickness)
            cv2.line(vis, (x2+w1-r, y2+r), (x2+w1+r, y2-r), col, thickness)
    return vis


if __name__ == '__main__':
    import sys
    try: fn1, fn2 = sys.argv[1:3]
    except:
        fn1 = "test_images/melody_face_frontal.png"
        fn2 = "test_images/melody_test_2.png"
    try:
        n_pyr = int(sys.argv[3])
    except:
        n_pyr = 0
        
    print help_message

    img1 = cv2.imread(fn1, cv2.CV_LOAD_IMAGE_COLOR)
    img2 = cv2.imread(fn2, cv2.CV_LOAD_IMAGE_COLOR)
    
    for i in range(n_pyr):
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
  
    grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    start = clock()

    surf = cv2.SURF(1000, _upright=True)
    
    kp2, desc2 = surf.detect(grey2, None, False)
    desc2.shape = (-1, surf.descriptorSize())

    kp1, desc1 = surf.detect(grey1, None, False)
    desc1.shape = (-1, surf.descriptorSize())
    print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))
    
    #kp1, desc1 = generate_samples(kp1, desc1)
    warped_samples = warp_target(grey1)
    
    for warp in warped_samples:
        cv.ShowImage("Warped", warp)
        cv.WaitKey(0)
    

    def match_and_draw(match, r_threshold):
        m = match(desc1, desc2, r_threshold)
        matched_p1 = np.array([kp1[i].pt for i, j in m])
        matched_p2 = np.array([kp2[j].pt for i, j in m])
        try:
            H, status = cv2.findHomography(matched_p1, matched_p2, cv2.LMEDS, 5.0)
            #H, status = cv2.findHomography(matched_p1, matched_p2, cv2.RANSAC, 5.0)
        except:
            H = None
            status = [1]*len(kp1)
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        vis = draw_match(img1, img2, matched_p1, matched_p2, status, H)
        return vis

    #print 'bruteforce match:',
    #vis_brute = match_and_draw( match_bruteforce, 0.75 )
    print 'flann match:',
    vis_flann = match_and_draw( match_flann, 0.75 ) # flann tends to find more distant second
                                                   # neighbours, so r_threshold is decreased
    print "Elapsed time:", 1000 * (clock() - start)
    #cv2.imshow('find_obj SURF', vis_brute)
    cv2.imshow('Find Object SURF Flann', vis_flann)
    cv2.waitKey()
