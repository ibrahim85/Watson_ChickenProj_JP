# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:36:15 2019

@author: jplineb
"""

import json
import pandas as pd
import requests
from PIL import Image, ImageDraw
import numpy
import urllib.request
import cv2

with open('export.json') as json_file:
    data = json.load(json_file)
    
columns= ["External_ID","Dataset", "geometry","cords", "mask"]
df = pd.DataFrame(columns = columns ) 

##### Test Code ######
#External_ID = data[0]["External ID"]
#Dataset = data[0]["Dataset Name"]
#geometry = data[0]["Label"]["Egg"][0]
#######################

for x in data:
    listofpoints = []
    External_ID = x["External ID"]
    Dataset = x["Dataset Name"]
    geometry = x['Label']['Egg'][0]
    for alpha, gamma in geometry.items():
        for beta in gamma:
            xcord = beta['x']
            ycord = beta['y']
            tuplecord = (xcord,ycord)
            listofpoints.append(tuplecord)
    mask = x['Masks']
    df = df.append({"External_ID" : External_ID,
                            "Dataset" : Dataset,
                            "geometry" : geometry,
                            "cords" : listofpoints,
                            "mask" : mask}, ignore_index = True)
    
##### Test Code ######

testimg = r'C:\Users\jplineb\Desktop\chicken project\Clutch1_D12\IMG_0004.JPG'
#cords = df["cords"][0]
#
#img = Image.new('L', (5328, 4000))
#ImageDraw.Draw(img).polygon(cords, outline = 1, fill = 1)
#mask = numpy.array(img)
#
#
#img.paste(Image.open(testimg), (0,0), mask)

urllib.request.urlretrieve('https://faas-gateway.labelbox.com/function/mask-exporter?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsYWJlbElkIjoiY2p3OXBhZTVuY2x4ejA3OTU2dDAzM3pjMyIsImNsYXNzTmFtZSI6IkVnZyIsImlhdCI6MTU1OTIzNDA2OSwiZXhwIjoxNzE2OTE0MDY5fQ.kEEqhUmZCipP-sdwiSLIwlIUhWnshIFCDLKHSF1WbPc', '1.png')
img = cv2.imread(testimg)
mask = cv2.imread('1.png', 0)
res = cv2.bitwise_and(img,img, mask = mask)

cv2.imwrite('10.png', res)