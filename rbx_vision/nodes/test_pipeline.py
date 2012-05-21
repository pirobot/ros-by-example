#!/usr/bin/env python

import re
import os
import itertools

def createpipeline(*filters):
  def pipeline(input):
      piped_data = input
      for filter in filters:
          piped_data = filter[0](piped_data, *filter[1:])
      return piped_data
  return pipeline

def text_match(text, pattern):
  p = re.compile(pattern)
  if p.search(text):
      return pattern
  else:
      return ""

def n_letters(text, n=1):
  return text[:n]


if __name__ == "__main__":
  input = "Hello World!"

  # Create pipeline
  pipeline = createpipeline((text_match, "World"),
                            (n_letters, 3))

  # Feed the pipeline    
  results = pipeline(input)

  # Print out each answer as it comes through the pipeline    
  for title in results:
      print title