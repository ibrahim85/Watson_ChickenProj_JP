# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:05:54 2019

@author: jplineb
"""

import os
import cv2

wrkingdir = os.getcwd()
maskroot = './masks'

def cropdamask(fileloc):
    img = cv2.pyrDown(cv2.imread(fileloc, cv2.IMREAD_UNCHANGED))
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                  0, 255, cv2.THRESH_BINARY) #Converts Image to black and white
    # find contours and get the external one
    contours, hier = cv2.findContours(threshed_img,  cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
    
    cnt, area = contours[0], 0
    for c in contours: #finds the contour with the largest area
        a = cv2.contourArea(c)
        if a>area:
            cnt=c
            area=a
            print(area)
    for c in cnt:
        x,y,w,h = cv2.boundingRect(cnt) # calculates the bounding rectangle
        #The line below was commented out to prevent issues with the bounding box showing in code
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #Draws the bounding rectangle
    roi = img[y:y+h, x:x+w] #sets the region of interest to the bounding rectangle
    cv2.imwrite(fileloc, roi) #crops inputed image



for root, dirs, files in os.walk(maskroot): #lists all files in directory
    for file in files:
        fileloc = os.path.join(root, file)
        cropdamask(fileloc)
        