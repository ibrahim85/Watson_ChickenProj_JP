# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:08:12 2019

@author: jplineb
"""

import os
from ibm_watson import VisualRecognitionV3
from ibm_watson import ApiException
import json
import pandas as pd
import time


visual_recognition = VisualRecognitionV3('2018-03-19', iam_apikey='bt3CdZLpeCLhkmy7W91LrhPLxFCbyEkGKu_SuVaGITDu')


results = pd.DataFrame([], columns = ['actual_class', 'predicted_class', 'score_male', 'score_female', 'img_name'])
male_dir = './Train_Test_prelimrun/test/male/'
male_test = os.listdir(male_dir)
female_dir = './Train_Test_prelimrun/test/female/'
female_test = os.listdir(female_dir)

#add dir to each file
male_test = [male_dir + x for x in male_test]
female_test = [female_dir + x for x in female_test]

for z in male_test:
    with open(z, 'rb') as image_file:
            classes = visual_recognition.classify(image_file, threshold= '0', classifier_ids='ChickenEggv2_363779308').get_result()
            outputs = (classes['images'][0])
            predicted_img_class_watson = outputs['classifiers'][0]['classes'][0]['class']
            score_male = outputs['classifiers'][0]['classes'][1]['score']
            score_female = outputs['classifiers'][0]['classes'][0]['score']
            img_name = outputs['image']
            results = results.append({'actual_class': 'Male',
                                      'predicted_class': predicted_img_class_watson,
                                      'score_male':score_male,
                                      'score_female':score_female,
                                      'img_name': img_name},ignore_index = True)
for z in female_test:
    with open(z, 'rb') as image_file:
            classes = visual_recognition.classify(image_file, threshold= '0', classifier_ids='ChickenEggv2_363779308').get_result()
            outputs = (classes['images'][0])
            predicted_img_class_watson = outputs['classifiers'][0]['classes'][0]['class']
            score_male = outputs['classifiers'][0]['classes'][1]['score']
            score_female = outputs['classifiers'][0]['classes'][0]['score']
            img_name = outputs['image']
            results = results.append({'actual_class': 'Female',
                                      'predicted_class': predicted_img_class_watson,
                                      'score_male':score_male,
                                      'score_female':score_female,
                                      'img_name': img_name},ignore_index = True)    

    

results.to_csv('results.csv')  
    