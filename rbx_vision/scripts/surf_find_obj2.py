#!/usr/bin/env python

import numpy as np
import cv2
from cv2 import cv as cv
from common import anorm, clock
from functools import partial
import random
from math import sin, cos
import sys


help_message = '''SURF image match 

USAGE: surf_find_obj.py [ <image1> <image2> ]
'''

FLANN_INDEX_KDTREE = 4  # bug: flann enums are missing

flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)

class FindSURF():
    def __init__(self):
        self.n_warps = 1
        self.noise_var = 1
        
        #self.surf = cv2.SURF(200, _upright=True)
        self.feature_type = "SURF"
        self.surf_kp = cv2.FeatureDetector_create(self.feature_type)
        self.surf_desc = cv2.DescriptorExtractor_create(self.feature_type)
        
        target_file = "test_images/mona_lisa_face.png"
        test_file = "test_images/mona_lisa.png"
        
        #target_file = "/home/patrick/Downloads/orl_faces/s32/10.pgm"
        #test_file = "/home/patrick/Downloads/orl_faces/s32/5.pgm"
                
        self.target_image = cv2.imread(target_file, cv2.CV_LOAD_IMAGE_COLOR)
        self.test_image = cv2.imread(test_file, cv2.CV_LOAD_IMAGE_COLOR)
        
        target_grey = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)
        test_grey = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        
        #target_grey = cv2.GaussianBlur(target_grey, (5,5), 5.0)
        #test_grey = cv2.GaussianBlur(test_grey, (5,5), 5.0)
        start = clock()

        self.kp1 = self.surf_kp.detect(target_grey, None)
        self.kp1, self.desc1 = self.surf_desc.compute(target_grey, self.kp1)
        self.desc1.shape = (-1, 128)
    
        self.kp2 = self.surf_kp.detect(test_grey, None)
        self.kp2, self.desc2 = self.surf_desc.compute(test_grey, self.kp2)
        self.desc2.shape = (-1, 128)
    
        print 'img1 - %d features, img2 - %d features' % (len(self.kp1), len(self.kp2))
    
        #print 'bruteforce match:',
        #vis_brute = match_and_draw( match_bruteforce, 0.75 )
        #print 'flann match:',
        vis_flann = self.match_and_draw(self.match_flann, self.target_image, self.test_image, 0.6 ) # flann tends to find more distant second
                                                       # neighbours, so r_threshold is decreased
        print "Elapsed time:", 1000 * (clock() - start)
        #cv2.imshow('find_obj SURF', vis_brute)
        cv.NamedWindow("SURF Match", cv.CV_WINDOW_NORMAL)
        cv2.imshow('SURF Match', vis_flann)
        cv2.waitKey(0)
        
        #kp1, desc1 = generate_samples(kp1, desc1)
#        warped_samples = self.warp_target(target_grey)
#        
#        for sample in warped_samples:
#            sample_array = np.array(sample, dtype=np.uint8)
#            sample_color = cv2.cvtColor(sample_array, cv2.COLOR_GRAY2BGR)
#            self.kp1, self.desc1 = self.surf.detect(sample_array, None, False)
#            self.desc1.shape = (-1, self.surf.descriptorSize())
#            print 'img1 - %d features, img2 - %d features' % (len(self.kp1), len(self.kp2))
#            vis_flann = self.match_and_draw(self.match_flann, sample_color, self.test_image, 0.75 )
#            cv2.imshow('Find Warped Object SURF Flann', vis_flann)
#            cv2.waitKey(0)
        
    def match_and_draw(self, match, target, test, r_threshold):
            m = match(self.desc1, self.desc2, r_threshold)
            matched_p1 = np.array([self.kp1[i].pt for i, j in m])
            matched_p2 = np.array([self.kp2[j].pt for i, j in m])
#            try:
#                H, status = cv2.findHomography(matched_p1, matched_p2, cv2.LMEDS, 5.0)
#                H, status = cv2.findHomography(matched_p1, matched_p2, cv2.RANSAC, 5.0)
#            except:
#                H = None
#                status = [1]*len(self.kp1)
            H = None
            status = [1]*len(self.kp1)
            print '%d / %d  inliers/matched' % (np.sum(status), len(status))
            vis = self.draw_match(target, test, matched_p1, matched_p2, status, H)
            return vis
    

    def warp_target(self, img):
        samples = list()
        warp = cv.CreateMat(2, 3, cv.CV_32FC1)    
        ran = random.Random()
        sample = cv.CloneMat(cv.fromarray(img))
        
        noise = cv.CloneMat(sample)
        
        cv.Zero(noise)
        cv.RandArr(cv.RNG(), noise, cv.CV_RAND_NORMAL, (0, 0, 0), (self.noise_var, self.noise_var, self.noise_var))
    
        #cv.Add(sample, noise, sample)
        
        #cv.EqualizeHist(sample, sample)
        
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
        
        param_theta = 0.5
        param_sin_cos = 0.5
        
        #sample = np.array(sample, dtype=np.uint8)
        for i in range(self.n_warps):
            result = cv.CloneMat(sample)
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
            cv.WarpAffine(mask, warped_mask, warp)
            cv.Copy(result, result, warped_mask)
            
#            r = ran.uniform(0, 180)
#            rotation_matrix = cv2.getRotationMatrix2D((sample.shape[1]/2, sample.shape[0]/2), r, 1.0)
#            result = cv2.warpAffine(sample, rotation_matrix, (sample.shape[1], sample.shape[0]), borderMode=cv2.BORDER_REPLICATE)
                        
    #            eig = cv.CreateImage(cv.GetSize(grey), 32, 1)
    #            temp = cv.CreateImage(cv.GetSize(grey), 32, 1)
    #            
    #            features = []
    #            
    #            features = cv.GoodFeaturesToTrack(result, eig, temp, self.max_count,
    #                    self.quality, self.good_feature_distance, mask=None, blockSize=3, useHarris=0, k=0.04)
    
    #        result_array = np.array(result, dtype=np.uint8)
    #        mask_array = np.array(warped_mask, dtype=np.uint8)
    #        keypoints, descriptors =  self.surf.detect(result_array, mask_array, False)
    #        
    #        displayed_keypoints = np.array([keypoints[i].pt for i in range(len(keypoints))])
    #        for (x1, y1) in np.int32(displayed_keypoints):
    #            cv.Circle(result, (int(x1), int(y1)), 2, (255, 255, 255, 0), cv.CV_FILLED, 8, 0)
    #       
            
            samples.append(result)
          
        return samples
    

    def match_bruteforce(self, desc1, desc2, r_threshold = 0.75):
        res = []
        for i in xrange(len(desc1)):
            dist = anorm(desc2 - desc1[i])
            n1, n2 = dist.argsort()[:2]
            r = dist[n1] / dist[n2]
            if r < r_threshold:
                res.append((i, n1))
        return np.array(res)
    
    def match_flann(self, desc1, desc2, r_threshold = 0.6):
        flann = cv2.flann_Index(desc2, flann_params)
        idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
        mask = dist[:,0] / dist[:,1] < r_threshold
        idx1 = np.arange(len(desc1))
        pairs = np.int32( zip(idx1, idx2[:,0]) )
        return pairs[mask]
    
    def draw_match(self, img1, img2, p1, p2, status = None, H = None):
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
    FS = FindSURF()

