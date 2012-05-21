#!/usr/bin/env python
from common import anorm, clock
from Orange.classification.svm import SVMLearner, SVMLearnerEasy
start = clock()
import cv2
from cv2 import cv as cv
import sys
import os, glob, errno
import numpy as np
import pickle
import hashlib
import Orange
import orngTree
from Orange.classification import svm, knn, tree, bayes
from Orange.evaluation import testing, scoring
import orange, orngTest, orngStat, orngTree, orngEnsemble
import warnings
import math
#print "Import time: ", 1000 * (clock() - start)

print "Running Face Classifier"

class FaceClassifier():
    def __init__(self):      
        n_eigenfaces = 4
        self.surf_faces_file = "surf_faces_" + str(n_eigenfaces) + ".pkl"
        self.classifier_file = "eigenclassifier_" + str(n_eigenfaces) + ".pkl"
        
        self.surf = cv2.SURF(200, _upright=True)
        
        have_eigenfaces = False
        classifier_trained = False
        show_eigenfaces = False
        
        self.cascade1= cv.Load("../models/haar_face_detector/haarcascade_frontalface_alt.xml")
        self.cascade2= cv.Load("../models/haar_face_detector/haarcascade_profileface.xml")
        
        #image_dir = "/home/patrick/Downloads/orl_faces"
        image_dir = "/home/patrick/Downloads/orl_faces"
        
        self.tab_file = "/home/patrick/tmp/faces_surf.tab"
        
        mean = None
        eigenvectors = None
        data = None
        
        # Do we already have the eigenfaces in a file?
        if os.path.isfile(self.surf_faces_file):
            data, labels, label_map, image_num, mean, eigenvectors, face_rows, face_cols = self.load_surf_faces()
        else:
            # Load the data and compute the eigenfaces.
            data, labels, label_map, image_num, face_rows, face_cols = self.load_data_surf(image_dir)
            mean, eigenvectors = self.compute_eigenfaces(data, labels, label_map, image_num, face_rows, face_cols, n_eigenfaces)
        
        #image_dir = "/home/patrick/Downloads/test_faces"
        #classifier = self.train_new_faces(image_dir, mean, eigenvectors, 92, 112)
        #self.classify_faces(data, classifier)
        
        # Have we loaded the data?        
        if data is None:
            data, labels, label_map, image_num, face_rows, face_cols = self.load_data_surf(image_dir)
            
         # Do we already have the classifier in a file?
        if os.path.isfile(self.classifier_file):
            classifier = self.load_classifier()
        else:
            self.train_classifier(data, labels, label_map, image_num, mean, eigenvectors)
            self.test_classifier(data, labels, label_map, image_num, mean, eigenvectors)

            #classifier = self.train_test_classifier(mean, eigenvectors)
        
#        samples = cv2.PCAProject(data, mean, eigenvectors)

#        for s in range(len(samples)):
#            label = classifier.predict(samples[s:s+1])
#            if labels[s] != label_map[int(label[0])]:
#                print labels[s], label_map[int(label[0])]
            
        # Test with one of the input images
        #test_image = cv2.imread("/home/patrick/Downloads/training_faces/Angelina_Jolie/Angelina_Jolie_0002.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
        #test_image = cv2.imread("/home/patrick/Downloads/training_faces/Monica_Lewinsky/Monica_Lewinsky_0003.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
        #test_image = cv2.imread("/home/patrick/Downloads/training_faces/Angelina_Jolie/Angelina_Jolie_0008.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
        test_image = cv2.imread("/home/patrick/Downloads/orl_faces/s8/7.pgm", cv2.CV_LOAD_IMAGE_GRAYSCALE)
        #test_image = cv2.imread("/home/patrick/Downloads/orl_faces/s35/2.pgm", cv2.CV_LOAD_IMAGE_GRAYSCALE)

        #self.classify_face(test_image, classifier, mean, eigenvectors, face_rows, face_cols, label_map)
        #self.export_data_tab(data, labels, label_map, mean, eigenvectors)
#        with warnings.catch_warnings():
#            warnings.filterwarnings("ignore",category=DeprecationWarning)
#            self.run_orange()
        
    def run_orange(self):
        start = clock()
        data = Orange.data.Table(self.tab_file)
        print "Data loading time:", 1000 * (clock() - start)

        print data.domain.classVar.values

        # Discretize the data
        #start = clock()
        #data = orange.Preprocessor_discretize(raw_data, method=orange.EntropyDiscretization())
        #print "Preprocessing time:", 1000 * (clock() - start)


        # Set up the learners
        Bayes = orange.BayesLearner(name="Bayes")
        Tree = tree.TreeLearner(name="Tree", same_majority_pruning=1, m_pruning=2)
        SVM = svm.SVMLearner(name="SVM", normalization=False, svm_type= SVMLearner.C_SVC, kernel_type=SVMLearner.Linear, C=0.03125, probability=False, eps=0.002)
        kNN = knn.kNNLearner(name="kNN", k=10)
        Forest = Orange.ensemble.forest.RandomForestLearner(name="Forest", trees=50)
        boosted_svm = orngEnsemble.BoostedLearner( SVM, name = "Boosted SVM", t=10)
        
        #learners = [tree, knn, svm]
        #learners = [bayes_class, knn_class, tree_class, svm_class, forest_class]
        learners = [SVM, kNN]
        
#        results = orngTest.crossValidation(learners, data, folds=10)
#        
#        # Output the results
#        print "Learner  CA     IS     Brier    AUC"
#        for i in range(len(learners)):
#            print "%-8s %5.3f  %5.3f  %5.3f  %5.3f" % (learners[i].name, \
#                orngStat.CA(results)[i], orngStat.IS(results)[i],
#                orngStat.BrierScore(results)[i], orngStat.AUC(results)[i])
            
        start = clock()
        SVM_classifier = SVM(data)
        print "Training time:", 1000 * (clock() - start)
        
#        class_time_ave = 0
#        n_samples = 0
#        for sample in data:
#            start = clock()
#            predict = SVM_classifier(sample)
#            #print predict, ": ",
#            class_time_ave += (clock() - start)
#            n_samples += 1
#            actual = sample.getclass()
#            #print actual
#            if predict != actual:
#                print predict, actual
#                
#        print "Ave classification time:", 1000 * class_time_ave / n_samples
    
    def match_flann(self, desc1, desc2, r_threshold = 0.6):
        flann = cv2.flann_Index(desc2, flann_params)
        idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
        mask = dist[:,0] / dist[:,1] < r_threshold
        idx1 = np.arange(len(desc1))
        pairs = np.int32( zip(idx1, idx2[:,0]) )
        return pairs[mask]
    
    def classify_face(self, test_image, classifier, mean, eigenvectors, face_width, face_height, label_map):
        #test_image = self.extract_face(test_image)
        #cv.EqualizeHist(test_image, test_image)
        test_image = cv2.equalizeHist(test_image)
        test_image = np.array(test_image, dtype=np.float32)
        test_image = cv2.resize(test_image, (face_width, face_height))
        cv2.normalize(test_image, test_image, 0, 1, cv2.NORM_MINMAX)
        image_vec = test_image.reshape(1, test_image.shape[0]*test_image.shape[1])
        projection = cv2.PCAProject(image_vec, mean, eigenvectors)

        label = classifier.predict(projection)
        print "Best Match:", label_map[int(label[0])]

    def export_data_tab(self, data, labels, label_map, mean, eigenvectors):
        samples = cv2.PCAProject(data, mean, eigenvectors)
        n_eigenvectors = samples.shape[1]
        n_samples = samples.shape[0]
        n_faces = len(label_map)
        
        data_fd = open(self.tab_file, 'w')
                
        # Create the header row for the tab-limited data file
        row = "name"
        for i in range(n_eigenvectors):
            row += "\tbin_" + str(i)
        row += "\n"
        data_fd.write(row)
        
        # Create the continous/discrete (c or d) label row
        row = "d"
        for i in range(n_eigenvectors):
            row += "\tc"
        row += "\n"
        data_fd.write(row)

        # Now write the "class" row indicating that the first column is the class label       
        row = "class"
        for i in range(n_eigenvectors):
            row += "\t"
        row += "\n"
        data_fd.write(row)
                
        for i in range(n_samples):
            # Get the name of the face
            row = labels[i]
            
            # Get the eigen coordindates
            sample = samples[i]

            for j in range(n_eigenvectors):
                row += "\t" + str(sample[j])
            row += "\n"
            data_fd.write(row)

        data_fd.close()
        
    def train_classifier(self, data, labels, label_map, image_num, mean, eigenvectors):
        print "Data Shape:", data.shape
        train_n = 5
        training_labels = list()
        for i in range(data.shape[0]):
            label = labels[i]
            image_index = image_num[i]
            if image_index in range(train_n):
                try:
                    training_data = np.vstack((training_data, data[i]))
                except:
                    training_data = data[i]
                training_labels.append(label_map.index(label))
        
        training_labels = np.array(training_labels, dtype=np.float32)

#        print training_labels.shape
#        print training_data.shape
            
        # First project the original data back onto the eigenvectors
        samples = cv2.PCAProject(training_data, mean, eigenvectors)
        self.mean_eigenvectors = [np.zeros_like(samples[0])] * len(label_map)
        n_samples = [0] * len(label_map)
        for i in range(samples.shape[0]):
            label = int(training_labels[i])
            self.mean_eigenvectors[label] += samples[i]
            n_samples[label] += 1
                    
        # Now reduce each class to its mean eigenvector projection
        for i in range(len(label_map)):
            self.mean_eigenvectors[i] = np.divide(self.mean_eigenvectors[i], float(n_samples[i]))

#        for i in range(len(label_map)):       
#            print self.mean_eigenvectors[i]
        
        
    def test_classifier(self, data, labels, label_map, image_num, mean, eigenvectors):
        test_range = xrange(0, 5)
        test_labels = list()
        test_data = dict()
        for i in range(len(label_map)):
            test_data[label_map[i]] = dict()
            
        for i in range(data.shape[0]):
            label = labels[i]
            if image_num[i] in test_range:
                try:
                    test_data[label][image_num[i]] = np.vstack((test_data[label][image_num[i]], data[i]))
                except:
                    test_data[label][image_num[i]] = data[i]
                                                                            
        # First project the original data back onto the eigenvectors
        keypoints = cv2.PCAProject(test_data['s1'][2], mean, eigenvectors)
        mean_eigenvector = np.zeros_like(keypoints[0])
        n_keypoints = keypoints.shape[0]
        for i in range(n_keypoints):
            mean_eigenvector += keypoints[i]
            
        mean_eigenvector = np.divide(mean_eigenvector, float(n_keypoints))
           
        min_distance = 10000
        best_match = 'None'
                
        for i in range(len(self.mean_eigenvectors)):
            distance = np.linalg.norm(mean_eigenvector - self.mean_eigenvectors[i])
            if distance < min_distance:
                min_distance = distance
                best_i = i
                best_match = label_map[i]
        
        print "Best match:", best_match
        print "Min dist:", min_distance
        print "Best i:", best_i
        print np.linalg.norm(mean_eigenvector - self.mean_eigenvectors[best_i])

        # Find the nearest trained mean vector to each test mean vector
#        for i in range(len(label_map)):       
#            print mean_eigenvectors[i]
    
    
    def load_surf_faces(self):
        input = open(self.surf_faces_file, 'rb')
        eigenfaces = pickle.load(input)
        input.close()
        data = eigenfaces['data']
        labels = eigenfaces['labels']
        label_map = eigenfaces['label_map']
        image_num = eigenfaces['image_num']
        mean = eigenfaces['mean']
        eigenvectors = eigenfaces['eigenvectors']
        face_rows = eigenfaces['face_rows']
        face_cols = eigenfaces['face_cols']
        return data, labels, label_map, image_num, mean, eigenvectors, face_rows, face_cols
            
    def compute_eigenfaces(self, data, labels, label_map, image_num, face_rows, face_cols, n_eigenvectors):
        print "Computing PCA using", n_eigenvectors, "eigenvectors..."
        mean, eigenvectors = cv2.PCACompute(data, np.mean(data, axis=0).reshape(1, -1), maxComponents=n_eigenvectors)
        print "Eigenvectors shape: ", eigenvectors.shape
        print "Done!"
        
        result = {'data': data, 'labels': labels, 'label_map': label_map, 'image_num': image_num, 'mean': mean, 'eigenvectors': eigenvectors, 'face_rows': face_rows, 'face_cols': face_cols}
        output = open(self.surf_faces_file, 'wb')
        pickle.dump(result, output)
        output.close()
        
        return mean, eigenvectors
    
    def load_data_surf(self, image_dir, face_width=100, face_height=100):
        print "Loading images..."
        cv.NamedWindow("Face")
        #data_file = '/home/patrick/tmp/face_vectors.tab'
        data_array = None
        
        # Get the images and store them in a data array
        n_people = 0
        n_keypoints = 0
        labels = list()
        label_map = list()
        image_num = list()

        for person in self.custom_listdir(image_dir):
            label_map.append(person)
            n_people += 1
            image_files = glob.glob(image_dir + "/" + person + "/*")
            image_files.sort()
            image_index = 0
            for image_file in image_files:
                image = cv2.imread(image_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                #image = self.extract_face(image)
                if image is None:
                    continue
                #cv.EqualizeHist(image, image)
                image = cv2.equalizeHist(image)
                image = np.array(image, dtype=np.uint8)
                image = cv2.resize(image, (face_width, face_height))
                kp, desc = self.surf.detect(image, None, False)
                desc.shape = (-1, self.surf.descriptorSize())
                n_keypoints += len(kp)
                
#                image = np.array(image, dtype=np.float32)
#                cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
#                kp_color = (0, 0, 0)
#                displayed_kp = np.array([kp[i].pt for i in range(len(kp))])
#                for (x1, y1) in np.int32(displayed_kp):
#                    cv2.circle(image, (x1, y1), 2, kp_color, -1)
#                
#                image_scaled = cv2.resize(image, (image.shape[0]*2, image.shape[1]*2))
#                cv2.imshow("Face", image_scaled)
#                cv2.waitKey()
                
                for i in range(len(kp)):
                    desc_vec = desc[i].reshape(1, 64)
                    #np.set_printoptions(suppress=False,linewidth=2000)
                    try:
                        data_array = np.vstack((data_array, desc_vec))
                    except:
                        data_array = desc_vec
                    labels.append(person)
                    image_num.append(image_index)
                    
                image_index += 1

        print n_keypoints, "keypoints across", n_people, "different people."
        return data_array, labels, label_map, image_num, face_height, face_width

    def custom_listdir(self, path):
        """
        Returns the content of a directory by showing directories first
        and then files by ordering the names alphabetically
        """
        dirs = sorted([d for d in os.listdir(path) if os.path.isdir(path + os.path.sep + d)])
        dirs.extend(sorted([f for f in os.listdir(path) if os.path.isfile(path + os.path.sep + f)]))
    
        return dirs
    
    def extract_face(self, image):
        min_size = (20, 20)
        haar_scale = 1.2
        min_neighbors = 2
        haar_flags = 0
        image_mat = cv.fromarray(image)
        if(self.cascade1):
            faces = cv.HaarDetectObjects(image_mat, self.cascade1, cv.CreateMemStorage(0),
                                         haar_scale, min_neighbors, haar_flags, min_size)
            if not faces and self.cascade2:
                faces = cv.HaarDetectObjects(image_mat, self.cascade2, cv.CreateMemStorage(0),
                                         haar_scale, min_neighbors, haar_flags, min_size)
            if faces:
                ((x, y, w, h), n) = faces[0]
                face_rect = (x, y, w, h)
                return cv.GetSubRect(image_mat, face_rect)
            else:
                return None

class SVM():
    def __init__(self, nclasses):
        self.model = cv2.SVM()
        self.nclasses = nclasses

    def train(self, samples, responses):
        params = dict( kernel_type = cv2.SVM_RBF, 
                       svm_type = cv2.SVM_C_SVC,
                       gamma = 0.0001,
                       C = self.nclasses )

#        params = dict( kernel_type = cv2.SVM_LINEAR, 
#                       svm_type = cv2.SVM_C_SVC,
#                       C = self.nclasses )

        print "N Classes:", self.nclasses
        
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )
    
class KNearest():
    def __init__(self, nclasses):
        self.model = cv2.KNearest()
        self.nclasses = nclasses
        
    def train(self, samples, responses):
        self.model.train(samples, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.find_nearest(samples, k = 10)
        return results.ravel()
    
class RTrees():
    def __init__(self, nclasses):
        self.model = cv2.RTrees()
        self.nclasses = nclasses

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        var_types = np.array([cv2.CV_VAR_NUMERICAL] * var_n + [cv2.CV_VAR_CATEGORICAL], np.uint8)
        #CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));
        params = dict(max_depth=10 )
        self.model.train(samples, cv2.CV_ROW_SAMPLE, responses, varType = var_types, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )
    
class ANN():
    def __init__(self, nclasses):
        self.model = cv2.ANN_MLP()
        self.nclasses= nclasses

    def train(self, samples, responses):
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.nclasses)

        layer_sizes = np.int32([var_n, 150, 150, self.nclasses])
        self.model.create(layer_sizes)
        
        # CvANN_MLP_TrainParams::BACKPROP,0.001
        params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 500, 0.01),
                       train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP, 
                       bp_dw_scale = 0.001,
                       bp_moment_scale = 0.0 )
        self.model.train(samples, np.float32(new_responses), None, params = params)
        
    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n * self.nclasses, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n) * self.nclasses )
        new_responses[resp_idx] = 1
        return new_responses

    def predict(self, samples):
        output = cv.CreateMat(samples.shape[0], self.nclasses, cv.CV_32FC1)
        output = np.array(output, dtype=np.float32)
        self.model.predict(samples, output)
        np.set_printoptions(threshold=np.nan)
        return output.argmax(-1)
    
    
if __name__ == "__main__":
    FaceClassifier()

        
