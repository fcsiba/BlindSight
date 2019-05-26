#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:32:47 2019

@author: rehan
"""

from imageai.Detection import ObjectDetection

import os
print(os.getcwd())
os.chdir('Documents/FYP/Project')
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


#custom_objects = detector.CustomObjects(person=True)

files=os.listdir('in/')[1:]
results={}


for f in files:
    print(f)
    detections, extracted_images = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "in/"+f), output_image_path=os.path.join(execution_path , "out/"+f), extract_detected_objects=True)

    for eachObject in detections:
        results[eachObject["name"]]=eachObject["percentage_probability"]


