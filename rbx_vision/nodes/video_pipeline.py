#!/usr/bin/env python

import roslib
roslib.load_manifest('pi_video_tracker')
import rospy
from ros2opencv2 import ROS2OpenCV2
import sys
import cv
import cv2
from sensor_msgs.msg import Image, RegionOfInterest
import re
import os
import itertools
import numpy as np

class ProcessVideo(ROS2OpenCV2):
    def __init__(self, node_name):
        ROS2OpenCV2.__init__(self, node_name)
        
        self.node_name = node_name
        self.grey = None
        self.detect_box = (1, 2, 3, 4)
        
        filters = [(self.grey_scale,), 
                   ('if', self.detect_box),
                   (self.blur, cv.CV_GAUSSIAN, 15, 0, 7.0),
                   (self.equalize,)]
                
        #filters.remove((self.blur, cv.CV_GAUSSIAN, 15, 0, 7.0))
        
        self.pipeline = self.create_pipeline(*filters)
        
        
#        self.pipeline = self.create_pipeline((if_pipe, 'not self.detect_box'),
#                                             (self.face_detect, 1),
#                                             (self.extract_features, gftt_params),
#                                             (self.extract_descriptors, surf_params),
#                                             (self.track_features, lk_params))
        
    def process_image(self, cv_image):            
        # Convert the image to a numpy array
        #frame = np.array(cv_image, dtype=np.uint8)
        
        result = self.pipeline(cv_image)
        #result = self.filter(self.equalize, self.filter(self.grey_scale, cv_image))
        #result = self.equalize(self.grey_scale(cv_image))
            
        return result
        
    def create_pipeline(self, *filters):
      def pipeline(frame):
          piped_frame = frame
          skip = False
          for filter in filters:
              if filter[0] == 'if':
                  if not filter[1]:
                      skip = True
                      continue
              else: 
                  if not skip:
                      piped_frame = filter[0](piped_frame, *filter[1:])
                      skip = False
          return piped_frame
      return pipeline

    
    def grey_scale(self, frame):
        grey = cv.CreateImage(cv.GetSize(frame), 8, 1)
        cv.CvtColor(frame, grey, cv.CV_BGR2GRAY)
        return grey
        #return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def equalize(self, frame):
        cv.EqualizeHist(frame, frame)
        return frame
        #return cv2.equalizeHist(frame)
        
    def blur(self, frame, *args):
        cv.Smooth(frame, frame, *args)
        return frame
  
def main(args):
      ProcessVideo("image_pipeline")
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print "Shutting down image pipeline node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

