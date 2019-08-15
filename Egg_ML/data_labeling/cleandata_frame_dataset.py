# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:19:30 2019

@author: jplineb
"""

import pandas as pd
import os
import numpy as np

df_clutch2 = pd.read_csv('./old_csvs/train_test.csv')
df_clutch2.drop(df_clutch2.columns[df_clutch2.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
df_clutch2['clutch'] = 2
def getnewfilepath(row):
    oldpath = row['Filepath']
    try:
        filename = oldpath.split('b/')[1]
        day = row['Day']
        if day == 12:
            newfold = 'Clutch2_D12/'
        if day==14:
            newfold = 'Clutch2_D14/'
        if day==18:
            newfold = 'Clutch2_D18/'
        newfilepath = './Cropped_Egg_images/' + newfold + filename
        return newfilepath
    except:
        return np.nan
        

df_clutch2["Crp_Filepath"] = df_clutch2.apply(getnewfilepath, axis =1)

df_clutch1 = pd.read_csv('chick_sexes_clutch1.csv')
df_clutch1 = df_clutch1.dropna()
df_clutch1['clutch'] = 1
df_clutch1['Day'] = 18
df_clutch2 = df_clutch2.rename(columns={'egg num':'egg_number'})
df_clutch1 = df_clutch1.rename(columns={'Sex 5.1.19':'sex'})
def createpath(row):
    egg_num = row['egg_number']
    fold = './Cropped_Egg_images/Clutch1_D' + str(row['Day'])
    filename = '/egg' + str(egg_num) + '.JPG'
    fullpath = fold + filename
    return fullpath

df_clutch1["Crp_Filepath"] = df_clutch1.apply(createpath, axis = 1)

df_all = df_clutch1.append(df_clutch2, ignore_index = True)
df_all = pd.DataFrame(df_all, columns = ['Crp_Filepath','clutch', 'Day', 'egg_number','sex'])
df_clean = df_all.dropna().reset_index()

df_clean.to_csv('cropped_test_train.csv')