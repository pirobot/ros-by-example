#!/usr/bin/env python
from common import anorm, clock
start = clock()
import cv2
from cv2 import cv as cv
import sys
import os, glob, errno
import numpy as np
from Orange.data import Table
from Orange.classification import svm, knn, tree, bayes
from Orange.evaluation import testing, scoring
import orange, orngTest, orngStat, orngTree, orngEnsemble
from Orange.classification.svm import SVMLearner, SVMLearnerEasy
print "Import time: ", 1000 * (clock() - start)

hist_size = 180
hist_range = [0, hist_size]
n_hist_bins = 180

class ColorClassifier():
    def __init__(self):
        cv.NamedWindow("Image", 0)
        cv.NamedWindow("Histogram", cv.CV_WINDOW_NORMAL)
        self.hist_image = cv.CreateImage((hist_size, n_hist_bins), 8, 1)
        
        self.smin = 1 #31
        self.vmin = 1 #41
        self.vmax = 254 #255
        
        cv.NamedWindow("Parameters", 0)
        cv.CreateTrackbar("Saturation", "Parameters", self.smin, 255, self.set_smin)
        cv.CreateTrackbar("Min Value", "Parameters", self.vmin, 255, self.set_vmin)
        cv.CreateTrackbar("Max Value", "Parameters", self.vmax, 255, self.set_vmax)

        start = clock()
        raw_data = Table("/home/patrick/tmp/training_histograms.tab")
        print "Data loading time:", 1000 * (clock() - start)

        print raw_data.domain.classVar.values

        # Discretize the data
        start = clock()
        data = orange.Preprocessor_discretize(raw_data, method=orange.EntropyDiscretization())
        print "Preprocessing time:", 1000 * (clock() - start)


        # Set up the learners
        #Bayes = orange.BayesLearner(name="Bayes")
        #DecisionTree = tree.TreeLearner(same_majority_pruning=1, m_pruning=2, name="Decision Tree")
        SVM = svm.SVMLearner(name="SVM")
        svm_class = svm.SVMLearner(normalization=True, svm_type= SVMLearner.C_SVC, kernel_type=SVMLearner.Linear, C=0.12, probability=False, eps=0.002)

        #kNN = knn.kNNLearner(k=10, name="kNN")
        #Forest = Orange.ensemble.forest.RandomForestLearner(trees=50, name="forest")
        #BostedSVM = orngEnsemble.BoostedLearner(svm, t=10, name="Boosted SVM")
        
        start = clock()
        svm_classifier = SVM(data)
        print "Training time:", 1000 * (clock() - start)
        
        learners = [SVM]

        start = clock()
        results = orngTest.crossValidation(learners, data, folds=5)
    
        # Output the results
        print "Learner  CA     IS     Brier    AUC"
        for i in range(len(learners)):
            print "%-8s %5.3f  %5.3f  %5.3f  %5.3f" % (learners[i].name, \
            orngStat.CA(results)[i], orngStat.IS(results)[i],
            orngStat.BrierScore(results)[i], orngStat.AUC(results)[i])
            
        print "Validation time:", 1000 * (clock() - start)

        class_time_ave = 0
        n_samples = 0
        for sample in data:
            start = clock()
            predict = svm_classifier(sample)
            class_time_ave += (clock() - start)
            n_samples += 1
            actual = sample.getclass()
            if predict != actual:
                print predict, actual
                
        print "Ave classification time:", 1000 * class_time_ave / n_samples


        # obtain class distribution
#        c = [0] * len(data.domain.classVar.values)
#        for e in data:
#            c[int(e.getclass())] += 1
#        print "Instances: ", len(self.data), "total"
#        for i in range(len(data.domain.classVar.values)):
#            print self.data.domain.classVar.values[i], ":", c[i]
#        print data[0].getclass()
        os._exit(0)
        
    def set_smin(self, pos):
        self.smin = pos
        
    def set_vmin(self, pos):
        self.vmin = pos
        
    def set_vmax(self, pos):
       self.vmax = pos
        
    def get_histogram(self, src_image):
        self.hsv_image = cv.CloneMat(src_image)
        #self.mask = cv.CreateMat(src_image.rows, src_image.cols, cv.CV_8UC1)
        
        # Blur the image
        cv.Smooth(src_image, src_image, smoothtype=cv.CV_GAUSSIAN, param1=5)
        
        cv.CvtColor(src_image, self.hsv_image, cv.CV_BGR2HSV)
        
        hsv_array = np.array(self.hsv_image, dtype=np.uint8)
        self.mask = cv2.inRange(hsv_array, np.array((0., self.smin, self.vmin)), np.array((180., 255., self.vmax)))

        #cv.InRange(self.hsv_image, cv.fromarray((0., self.smin, self.vmin)), cv.fromarray   ((180., 255., self.vmax)), self.mask)

        self.hue_image = cv.CreateMat(src_image.rows, src_image.cols, cv.CV_8UC1)
        self.sat_image = cv.CreateMat(src_image.rows, src_image.cols, cv.CV_8UC1)
        self.val_image = cv.CreateMat(src_image.rows, src_image.cols, cv.CV_8UC1)
        self.norm_val_image = cv.CreateMat(src_image.rows, src_image.cols, cv.CV_8UC1)
        
        cv.Split(self.hsv_image, self.hue_image, self.sat_image, self.val_image, None)
        
        hist = cv.CreateHist([n_hist_bins], cv.CV_HIST_ARRAY, [hist_range], 1)
        
        #masked = cv.CreateMat(src_image.rows, src_image.cols, src_image.type)
        #cv.Copy(src_image, masked, cv.fromarray(self.mask))
        #cv.ShowImage("Image", masked)

        cv.CalcArrHist([self.hue_image], hist, mask=cv.fromarray(self.mask))
        hist.bins[0] = 0
        hist.bins[n_hist_bins - 1] = 0
        #cv.NormalizeHist(hist, 255)

        (min_value, max_value, _, _) = cv.GetMinMaxHistValue(hist)
        #cv.Scale(hist.bins, hist.bins, float(self.hist_image.height) / max_value, 0)
        if max_value != 0:
            cv.Scale(hist.bins, hist.bins, 255. / max_value, 0)
            
        cv.Set(self.hist_image, cv.ScalarAll(255))
        bin_w = round(float(self.hist_image.width) / n_hist_bins)

        for i in range(n_hist_bins):
            #print hist.bins[i]
            cv.Rectangle(self.hist_image, (int(i * bin_w), self.hist_image.height),
                         (int((i + 1) * bin_w), self.hist_image.height - cv.Round(hist.bins[i])),
                         cv.ScalarAll(0), -1, 8, 0)
       
        #cv.ShowImage("Histogram", self.hist_image)
        #cv.WaitKey(0)
        
        return hist        

if __name__ == "__main__":
    # Load the source image.
    if len(sys.argv) > 1:
        src_image = cv.GetMat(cv.LoadImage(sys.argv[1], 1))
    else:
#        train = True
#        if train:
#            image_dir = '/home/patrick/Dropbox/Robotics/Color_Vision/training_images/'
#            #data_file = '/home/patrick/Dropbox/Robotics/Color_Vision/histograms.tab'
#            data_file = '/home/patrick/tmp/training_histograms.tab'
#        else:
#            image_dir = '/home/patrick/Dropbox/Robotics/Color_Vision/test_images/'
#            #data_file = '/home/patrick/Dropbox/Robotics/Color_Vision/histograms.tab'
#            data_file = '/home/patrick/tmp/test_histograms.tab'

        image_dir = list()
        image_dir.append("/home/patrick/Dropbox/Robotics/Color_Vision/training_images/")
        image_dir.append("/home/patrick/Dropbox/Robotics/Color_Vision/test_images/")
        data_file = '/home/patrick/tmp/training_histograms.tab'
            
        #file = '/home/patrick/Dropbox/Robotics/Color_Vision/training_images/green/0005.jpg'
        #src_image = cv.GetMat(cv.LoadImage(file, 1))
        CC = ColorClassifier()
        
        data_fd = open(data_file, 'w')
                
        # Create the header row for the tab-limited data file
        row = "color"
        for i in range(n_hist_bins):
            row += "\tbin_" + str(i)
        row += "\n"
        data_fd.write(row)
        
        # Create the continous/discrete (c or d) label row
        row = "d"
        for i in range(n_hist_bins):
            row += "\tc"
        row += "\n"
        data_fd.write(row)

        # Now write the "class" row indicating that the first column is the class label       
        row = "class"
        for i in range(n_hist_bins):
            row += "\t"
        row += "\n"
        data_fd.write(row)
        
        colors = ['blue', 'red', 'yellow', 'green', 'purple', 'orange']
        
        for k in range(2):
            for class_dir in os.listdir(image_dir[k]):
                if not class_dir in colors:
                    continue
                print class_dir
                for image_file in os.listdir(image_dir[k] + '/' + class_dir):
                    print image_file
                    #if image_file != '0002.jpg':
                        #continue
                    hist_string = ""
                    image = cv.GetMat(cv.LoadImage(image_dir[k] + '/' + class_dir + '/' + image_file, 1))
                    histogram = CC.get_histogram(image)
                    row = class_dir
                    for i in range(n_hist_bins):
                        row += "\t" + str(histogram.bins[i])
                    row += "\n"
                    data_fd.write(row)

        data_fd.close()
        
        print "Finished!"
