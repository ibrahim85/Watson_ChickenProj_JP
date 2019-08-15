# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:28:30 2019

@author: jplineb
"""

import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
df = pd.read_csv('cropped_test_train.csv')
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


df_train, df_test = train_test_split(df, test_size=.2, stratify=df.sex)

def copytrain(row):
    sourcefile = row['Crp_Filepath']
    sex = row['sex']
    newfiledir = './Watson/Train_Test_prelimrun/train/' + sex + '/'
    if not os.path.exists(newfiledir):
        os.makedirs(newfiledir)
    newdest = newfiledir + sourcefile.split('/')[3]
    shutil.copy(sourcefile,newdest)
    return (None)

def copytest(row):
    sourcefile = row['Crp_Filepath']
    sex = row['sex']
    newfiledir = './Watson/Train_Test_prelimrun/test/' + sex + '/'
    if not os.path.exists(newfiledir):
        os.makedirs(newfiledir)
    newdest = newfiledir + sourcefile.split('/')[3]
    shutil.copy(sourcefile,newdest)
    return (None)



df_train.apply(copytrain, axis = 1)    
df_test.apply(copytest, axis = 1) 













#def copytrain(dataframe):
#    df2 = df_train.loc[df_train.sex=='Male']
#    for x in df2.index:
#        filepath = df2['Filepath'][x]
#        sex = df2['sex'][x]
        