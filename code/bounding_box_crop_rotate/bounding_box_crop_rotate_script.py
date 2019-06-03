# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:37:01 2019

@author: jplineb
"""

import os
import cv2



wrkingdir = os.getcwd()

img = cv2.pyrDown(cv2.imread("egg30.jpg", cv2.IMREAD_UNCHANGED))

ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                  0, 255, cv2.THRESH_BINARY) #sets the image to gray scale (thresshold was white vs black here)

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
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #Drawis the bounding rectangle

img2 = cv2.resize(img, (1438, 1080))
cv2.imshow("contours", img2)
 

roi = img[y:y+h, x:x+w] #sets the region of interest to the bounding rectangle
cv2.imwrite('image_crop.jpg', roi) #crops inputed image

cv2.waitKey(0)

cv2.destroyAllWindows()

