#!/usr/bin/env python


""" svm_test.py - Version 1.0 2012-02-28

    Test the OpenCV SVM function for outlier detection
    
"""
import cv2.cv as cv
import cv2
import numpy as np
from time import clock, time

class SVMTest():
    def __init__(self):
        model = OneClassSVM()
        
        samples = np.array(0.3 * np.random.randn(50, 2), dtype = np.float32)
        
        start = time()
        model.train(samples)
        duration = time() - start
        
        print "Duration:", duration
        
        #test_rate  = np.mean(model.predict(samples) == 1)
        print model.predict(samples)
        
        #print 'Test rate: %f' % test_rate*100

        
class OneClassSVM():
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples):
        params = dict( kernel_type = cv2.SVM_LINEAR, 
                       svm_type = cv2.SVM_ONE_CLASS, nu = 0.1 )
        responses = np.array(np.empty(samples.shape), dtype = np.float32)
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )


if __name__ == '__main__':
    SVMTest()
    