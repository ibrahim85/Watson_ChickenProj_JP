# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:04:23 2019

@author: jplineb
"""

from PIL import Image, ExifTags
import pandas as pd
import os

df_ChickenTimes = pd.read_excel('Chicken_egg_time.xlsx')
df_ChickenTimes['Time'] = df_ChickenTimes['Time'].astype('str') # converts object to string to parse for datatime conversion
df_ChickenTimes['TimeDT'] = pd.to_datetime(df_ChickenTimes['Time'],
               format = '%H:%M:%S') # Converts the new string object to a readable date time format
df_ChickenTimes['TimeInt'] = (df_ChickenTimes['TimeDT'].dt.hour * 3600 +
               df_ChickenTimes['TimeDT'].dt.minute * 60 +
               df_ChickenTimes['TimeDT'].dt.second) #Converts the time to seconds
# Calculate time Differences
df_ChickenTimes['Delta'] = df_ChickenTimes.TimeInt.diff(periods=-1) # gets the time delta to find out times for possible images
##

# Create a list of possible times
def posstimes(row):
    starttime = row['TimeInt']
    try:
        delta = row['Delta']
        if delta >=0:
            delta = -60
        listofpostimes = list(range(int(starttime),int(starttime-delta)))
    except:
        delta = -60
        listofpostimes = list(range(int(starttime),int(starttime-delta)))
    return (listofpostimes)

df_ChickenTimes['poss_times'] = df_ChickenTimes.apply(posstimes, axis = 1) 
##

# Test for creating a dictionsary of Exif tag items from the meta data of the image #
#img = Image.open("./data/d12/rgb/IMG_0001.jpg")
#exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
######################################################################################

ThewholeDB = []
pathsofinterest = ['./data/d12/rgb', './data/d14/rgb', './data/d18/rgb']
daysofinterest = ['d12','d14','d18']
for pathdest in pathsofinterest:
    for (dirpath, subfolders, filenames) in os.walk(pathdest):
        for x in filenames:
            ThewholeDB.append(os.path.join(os.path.relpath(dirpath),x))

# Extracts meta data from a filename list  for photos
def getmetadata(filenamelist, justtime = False):
    tupleslist = []
    for x in filenamelist:
        img = Image.open(x)
        exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
        if justtime:
            exif = exif['DateTime']
        tupleslist.append((x,exif))
    return(tupleslist)
########################################################
    
Filemetadata = getmetadata(ThewholeDB, justtime = True) #runs the function to grab meta data from all of the files


# Create dataframe from new list
dictlist = []
for x in Filemetadata:
     newlist = ({'day':int((x[0].split('\\')[1]).split('d')[1]), 'img_path':x[0],'time_taken':x[1].split(' ')[1]})
     dictlist.append(newlist)

df_photos = pd.DataFrame(dictlist, columns =['day','img_path', 'time_taken'])
df_photos['time_taken'] = df_photos['time_taken'].astype('str') # converts object to string to parse for datatime conversion
df_photos['TimeDT'] = pd.to_datetime(df_photos['time_taken'],
               format = '%H:%M:%S') # Converts the new string object to a readable date time format
df_photos['TimeInt'] = (df_photos['TimeDT'].dt.hour * 3600 +
               df_photos['TimeDT'].dt.minute * 60 +
               df_photos['TimeDT'].dt.second)

##

def getframes(row):
    filenames = []
    listofposstimes = row['poss_times']
    dayfortime = row['Day']
    df_specific = df_photos[(df_photos['TimeInt'].isin(listofposstimes)) & (df_photos['day']==dayfortime)]
    picpaths = df_specific.img_path.tolist()
    for x in picpaths:
        imgname = x.split('\\')[3]
        filenames.append(imgname)
        
    return (filenames)

df_ChickenTimes['Files'] = df_ChickenTimes.apply(getframes, axis = 1)

df_Clean = pd.DataFrame(df_ChickenTimes,columns = ['Day', "egg num", 'Notes', 'Files'])

df_Clean.to_excel('Chicken_Egg_Images.xlsx')
































