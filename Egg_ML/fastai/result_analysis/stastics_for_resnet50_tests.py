# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:34:54 2019

@author: jplineb
"""

import os
import pandas as pd
import numpy as np

df = pd.read_csv('DF_new_hyperparameters.csv')
df_mean = df.groupby('test_name').mean()
df_mean_agg = df.groupby(['test_name']).agg([np.average, lambda x: np.std(x)/np.sqrt(5)])

df_2 = pd.read_csv('DF_resnet18_7_1_0.3_modified_true.csv')

