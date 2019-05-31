# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:37:01 2019

@author: jplineb
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


wrkingdir = os.getcwd()

img = cv2.pyrDown(cv2.imread("egg30.jpg", cv2.IMREAD_UNCHANGED))

ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                  0, 255, cv2.THRESH_BINARY) #sets the image to gray scale (thresshold was white vs black here)

# find contours and get the external one
contours, hier = cv2.findContours(threshed_img,  cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

cnt, area = contours[0], 0
for c in contours:
    a = cv2.contourArea(c)
    if a>area:
        cnt=c
        area=a
        print(area)


for c in cnt:
    x,y,w,h = cv2.boundingRect(cnt)
    gamma = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#    ellipse = cv2.fitEllipse(c)
#    cv2.ellipse(img, ellipse, (0,255,0), 2)
#    
#    (x,y),radius = cv2.minEnclosingCircle(c)
#    center = (int(x),int(y))
#    radius = int(radius)
#    cv2.circle(img,center,radius,(0,255,0),2)

img2 = cv2.resize(img, (1438, 1080))
cv2.imshow("contours", img2)
 
cv2.waitKey(0)

cv2.destroyAllWindows()




##crop
def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0) 
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop

alpha = crop_minAreaRect(img,)



#
#def subimage(image, center, theta, width, height):
#
#   ''' 
#   Rotates OpenCV image around center with angle theta (in deg)
#   then crops the image according to width and height.
#   '''
#
#   # Uncomment for theta in radians
#   theta *= 180/np.pi
#
#   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)
#
#   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
#   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )
#
#   x = int( center[0] - width/2  )
#   y = int( center[1] - height/2 )
#
#   image = image[ y:y+height, x:x+width ]
#
#   return image
#
#width = int(rect[1][1] - rect[1][0])
#height = int(rect[0][1] - rect[0][1])
#alpha = width/2 + rect[1][0]
#beta = height/2 + rect[0][1]
#center = (alpha,beta)
#theta = int(rect[2])
#
#
#
#img = subimage(img, center, theta, width, height)
#
#cv2.imwrite('path.jpg', img)
#
#

################
#img = cv2.resize(img, (1438, 1080))
#cv2.imshow("crop", img)
# 
#cv2.waitKey(0)
#cv2.destroyAllWindows()
