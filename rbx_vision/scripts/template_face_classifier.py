#!/usr/bin/env python
from common import anorm, clock
import cv2
from cv2 import cv as cv
import sys
import os, glob, errno
import numpy as np
import warnings

print "Running the Face Classifier"

class FaceClassifier():
    def __init__(self):     
        self.cascade1= cv.Load("../models/haar_face_detector/haarcascade_frontalface_alt.xml")
        self.cascade2= cv.Load("../models/haar_face_detector/haarcascade_profileface.xml")
        
        image_dir = "/home/patrick/Downloads/orl_faces"
        
        self.tab_file = "/home/patrick/tmp/faces.tab"
        
        mean = None
        eigenvectors = None
        data = None
        
        # Load the data and compute the eigenfaces.
        data, labels, label_map, face_rows, face_cols = self.load_data(image_dir)

        # Display the mean face
        #print face_rows, face_cols
        cv.NamedWindow("Mean Face", cv.CV_WINDOW_NORMAL)
        cv2.imshow("Mean Face", mean.reshape(face_rows, face_cols))

        # Display the individual eigenfaces
        if show_eigenfaces:
            cv.NamedWindow("Eigen Face", cv.CV_WINDOW_NORMAL)
            for i in range(eigenvectors.shape[0]):
                eigen_face = eigenvectors[i].copy()
                cv2.normalize(eigen_face, eigen_face, 0, 1, cv2.NORM_MINMAX)
                cv2.imshow("Eigen Face", eigen_face.reshape(face_rows, face_cols))
                cv2.waitKey()
        
        #image_dir = "/home/patrick/Downloads/test_faces"
        #classifier = self.train_new_faces(image_dir, mean, eigenvectors, 92, 112)
        #self.classify_faces(data, classifier)
        
        # Have we loaded the data?        
        if data is None:
            data, labels, label_map, face_rows, face_cols = self.load_data(image_dir)
            
         # Do we already have the classifier in a file?
        if os.path.isfile(self.classifier_file):
            classifier = self.load_classifier()
        else:
            classifier = self.train_classifier(data, labels, label_map, mean, eigenvectors)
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
        #test_image = cv2.imread("/home/patrick/Downloads/training_faces/Adam_Sandler/Adam_Sandler_0003.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
        #test_image = cv2.imread("/home/patrick/Downloads/CroppedYale/yaleB03/yaleB03_P00A-005E+10.pgm", cv2.CV_LOAD_IMAGE_GRAYSCALE)


        self.reconstruct(test_image, mean, eigenvectors, face_rows, face_cols)
        self.classify_face(test_image, classifier, mean, eigenvectors, face_rows, face_cols, label_map)
        self.export_data_tab(data, labels, label_map, mean, eigenvectors)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            self.run_orange()
    
    def train_new_faces(self, data, mean, eigenvectors, face_width, face_height):
        pass
    
    def classify_face(self, test_image, classifier, mean, eigenvectors, face_width, face_height, label_map):
        test_image = self.extract_face(test_image)
        #cv.EqualizeHist(test_image, test_image)
        test_image = np.array(test_image, dtype=np.float32)
        #test_image = cv2.GaussianBlur(test_image, (5, 5), 3.0)
        test_image = cv2.resize(test_image, (face_width, face_height))
        cv2.normalize(test_image, test_image, 0, 1, cv2.NORM_MINMAX)
        image_vec = test_image.reshape(1, test_image.shape[0]*test_image.shape[1])
        projection = cv2.PCAProject(image_vec, mean, eigenvectors)

        label = classifier.predict(projection)
        print "Best Match:", label_map[int(label[0])]
        
    def reconstruct(self, test_image, mean, eigenvectors, face_width, face_height):
        cv.NamedWindow("Test Face", cv.CV_WINDOW_NORMAL)
        cv.NamedWindow("Extracted Face", cv.CV_WINDOW_NORMAL)
        cv2.imshow("Test Face", test_image)
        test_image = self.extract_face(test_image)
        #cv.EqualizeHist(test_image, test_image)
        test_image = np.array(test_image, dtype=np.float32)
        test_image = cv2.resize(test_image, (face_width, face_height))
        #test_image = cv2.pyrDown(test_image)
        cv2.normalize(test_image, test_image, 0, 1, cv2.NORM_MINMAX)
        cv2.imshow("Extracted Face", test_image)
        image_vec = test_image.reshape(1, test_image.shape[0]*test_image.shape[1])
        projection = cv2.PCAProject(image_vec, mean, eigenvectors)
        back_project = cv2.PCABackProject(projection, mean, eigenvectors)

        cv.NamedWindow("Reconstructed", cv.CV_WINDOW_NORMAL)
        cv2.imshow("Reconstructed", back_project.reshape(test_image.shape))
        cv2.waitKey()

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
    
    def train_classifier(self, data, labels, label_map, mean, eigenvectors):
        # First project the original data back onto the eigenvectors
        samples = cv2.PCAProject(data, mean, eigenvectors)
        responses = []
        for i in range(samples.shape[0]):
            response = label_map.index(labels[i])
            responses.append(response)
        responses = np.array(responses, dtype=np.float32)
                    
        rnd_state = np.random.get_state()
        np.random.shuffle(samples)
        np.random.set_state(rnd_state)
        np.random.shuffle(responses)
        
        nclasses = len(label_map)
        model = SVM(nclasses)
        #model = KNearest()
        #model = RTrees(nclasses)
        #model = ANN(nclasses)
        
        train_ratio = 0.7
        train_n = int(len(samples)*train_ratio)
        
        print "Training classifier...",
        start = clock()
        model.train(samples[:train_n], responses[:train_n])
        duration = clock() - start
        print "Done!"
        
        print "Training time: ", 1000 * (clock() - start)
        
#        print responses
#        map_label = dict((v,k) for k, v in label_map.iteritems())
#        
#        pred = model.predict(samples[0:1])
#        person = map_label[int(pred[0])]
#        print person
        
        train_rate = np.mean(model.predict(samples[:train_n]) == responses[:train_n])                                                                       
        test_rate  = np.mean(model.predict(samples[train_n:]) == responses[train_n:])
        
        print 'train rate: %f  test rate: %f' % (train_rate*100, test_rate*100)
        
        return model
    
    def load_eigenfaces(self):
        input = open(self.eigenfaces_file, 'rb')
        eigenfaces = pickle.load(input)
        input.close()
        data = eigenfaces['data']
        labels = eigenfaces['labels']
        label_map = eigenfaces['label_map']
        mean = eigenfaces['mean']
        eigenvectors = eigenfaces['eigenvectors']
        face_rows = eigenfaces['face_rows']
        face_cols = eigenfaces['face_cols']
        return data, labels, label_map, mean, eigenvectors, face_rows, face_cols
            
    def compute_eigenfaces(self, data, labels, label_map, face_rows, face_cols, n_eigenvectors):
        print "Computing PCA using", n_eigenvectors, "eigenvectors...",
        mean, eigenvectors = cv2.PCACompute(data, np.mean(data, axis=0).reshape(1, -1), maxComponents=n_eigenvectors)
        print eigenvectors.shape
        print "Done!"
        
        result = {'data': data, 'labels': labels, 'label_map': label_map, 'mean': mean, 'eigenvectors': eigenvectors, 'face_rows': face_rows, 'face_cols': face_cols}
        output = open(self.eigenfaces_file, 'wb')
        pickle.dump(result, output)
        output.close()
        
        return mean, eigenvectors
    
    def load_data(self, image_dir, face_width=100, face_height=100):
        print "Loading images..."
        data_array = None
        
        # Get the images and store them in a data array
        n_people = 0
        n_images = 0
        labels = list()
        label_map = list()

        for person in self.custom_listdir(image_dir):
            label_map.append(person)
            print person
            n_people += 1
            image_files = glob.glob(image_dir + "/" + person + "/*")
            image_files.sort()
            for image_file in image_files:
                image = cv2.imread(image_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                #image = self.extract_face(image)
                if image is None:
                    continue
                labels.append(person)
                n_images += 1
                cv.EqualizeHist(image, image)
                
                #image = cv2.equalizeHist(image)
                image = np.array(image, dtype=np.float32)
                #image = cv2.GaussianBlur(image, (5, 5), 3.0)
                #image = cv2.resize(image, (face_width, face_height))
                #image = cv2.pyrDown(image)
                cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
                cv2.imshow("Face", image)
                cv2.waitKey()
                data


        print n_images, "images loaded for", n_people, "different people."
        return data_array, labels, label_map, face_height, face_width

    def run_orange(self):
        start = clock()
        data = Orange.data.Table("/home/patrick/tmp/faces.tab")
        print "Data loading time:", 1000 * (clock() - start)

        print data.domain.classVar.values

        # Discretize the data
        #start = clock()
        #data = orange.Preprocessor_discretize(raw_data, method=orange.EntropyDiscretization())
        #print "Preprocessing time:", 1000 * (clock() - start)


        # Set up the learners
        bayes_class = orange.BayesLearner()
        tree_class = tree.TreeLearner(same_majority_pruning=1, m_pruning=2)
        svm_class = svm.SVMLearner(normalization=False, svm_type= SVMLearner.C_SVC, kernel_type=SVMLearner.Linear, C=0.03125, probability=False, eps=0.002)
        knn_class = knn.kNNLearner(k=10)
        forest_class = Orange.ensemble.forest.RandomForestLearner(trees=50, name="forest")
        boosted_svm_class = orngEnsemble.BoostedLearner(svm, t=10)
        boosted_svm_class.name = "Boosted SVM"
        
        bayes_class.name = "bayes"
        tree_class.name = "tree"
        svm_class.name = "SVM"
        knn_class.name = "kNN"
        #learners = [tree, knn, svm]
        #learners = [bayes_class, knn_class, tree_class, svm_class, forest_class]
        learners = [svm_class, knn_class]
        
        results = orngTest.crossValidation(learners, data, folds=10)
        
        # Output the results
        print "Learner  CA     IS     Brier    AUC"
        for i in range(len(learners)):
            print "%-8s %5.3f  %5.3f  %5.3f  %5.3f" % (learners[i].name, \
                orngStat.CA(results)[i], orngStat.IS(results)[i],
                orngStat.BrierScore(results)[i], orngStat.AUC(results)[i])
        

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
#        params = dict( kernel_type = cv2.SVM_RBF, 
#                       svm_type = cv2.SVM_C_SVC,
#                       gamma = 0.0001,
#                       C = self.nclasses )

        params = dict( kernel_type = cv2.SVM_LINEAR, 
                       svm_type = cv2.SVM_C_SVC,
                       C = self.nclasses )

        print "N Classes:", self.nclasses
        
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples] )
    
class KNearest():
    def __init__(self):
        self.model = cv2.KNearest()

    def train(self, samples, responses, nclasses):
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

        layer_sizes = np.int32([var_n, 100, 100, self.nclasses])
        self.model.create(layer_sizes)
        
        # CvANN_MLP_TrainParams::BACKPROP,0.001
        params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 300, 0.01),
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

        
