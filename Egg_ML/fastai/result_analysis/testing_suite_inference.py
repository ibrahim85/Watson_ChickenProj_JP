# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:06:33 2019

@author: jplineb
"""

import os
import pandas as pd
import numpy as np

df = pd.read_csv('DF_resnet18_epoch9.csv')

def fixtrans(row):
    test_name = row['test_name']
    transforms = test_name.split('_')[2]
    return transforms

def fixnorm(row):
    test_name = row['test_name']
    norm = test_name.split('_')[3]
    return norm

df['transforms'] = df.apply(fixtrans, axis = 1)
df['normalized'] = df.apply(fixnorm, axis = 1)
#df = df.drop(columns = ['split_nummax_error'])
df_mean = df.groupby('test_name').mean()
df_agg = df.groupby(['test_name']).agg([np.average, lambda x: np.std(x)/np.sqrt(5)])
df_agg_clean = pd.DataFrame(df, columns=['test_name', 'min_error'])
df_agg_clean = df_agg_clean.groupby(['test_name']).agg([np.average, lambda x: np.std(x)/np.sqrt(5)])

def findmaxmean(row):
    mean_error = row['min_error']['average']
    error_bar = row['min_error']['<lambda>']
    return (mean_error+(2*error_bar))

df_agg_clean['max_mean_error'] = df_agg_clean.apply(findmaxmean, axis =1)

# Use for writing results to Excel
#with pd.ExcelWriter('Test_suite_results_favs.xlsx') as writer:
#    df.to_excel(writer, sheet_name='All_Results')
#    df_mean.to_excel(writer, sheet_name='Means_grouped')
#    df_agg.to_excel(writer, sheet_name='Agg')