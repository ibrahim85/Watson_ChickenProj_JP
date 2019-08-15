# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:47:06 2019

@author: jplineb
"""

'''
This py file digestes the results from the watson outputs
'''

import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

df = pd.read_csv('results.csv')
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

y_true = df.actual_class.tolist()
y_pred = df.predicted_class.tolist()



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def calibpred(row):
    score_male = row.weighted_male
    score_female = row.weighted_female
    prediction = score_male*Cs[0] + score_female*Cs[1] + estim.intercept_
    #prediction = estim.predict(Xnew)
    return prediction

def mof(row):
    pred = row.calib_pred
    pred = round(pred)
    if pred == 1:
        sex = 'Male'
    if pred == 0:
        sex = 'Female'
    return sex

plot_confusion_matrix(y_true, y_pred,
                      classes = ['Female','Male'],
                      title = 'Watson initial results')

def cmfromcm(cm, classes, normalize=False, title=None,cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax




#### Model Calibration ####

def getweightedscoremale(row):
    score_male = row.score_male
    score_female = row.score_female
    total_score = score_male + score_female
    weight_male = score_male/total_score
    
    return (weight_male)

def getweightedscorefemale(row):
    score_male = row.score_male
    score_female = row.score_female
    total_score = score_male + score_female
    weight_female = score_female/total_score
    
    return (weight_female)

df['weighted_male'] = df.apply(getweightedscoremale, axis = 1)
df['weighted_female'] = df.apply(getweightedscorefemale, axis = 1)

# Change the male and female to represent 1 and 0
df['actual_class'] = df.actual_class.map({'Male':1,'Female':0})
df['predicted_class'] = df.predicted_class.map({'Male':1, 'Female':0})

# Create data frame for while loop
df_while = pd.DataFrame(df)
df_repeated = pd.DataFrame(df)

## Create Test and Train Set
calib_frac = .3
df_train, df_test = train_test_split(df, test_size=calib_frac, stratify=df.actual_class)

## creating list of train and test
y_dftest = df_test.actual_class.values
x_dftest = df_test.iloc[:, 5:7].values

y_dftrain = df_train.actual_class.values
x_dftrain = df_train.iloc[:, 5:7].values

## Model Calibration
estim = LinearRegression(fit_intercept= False)
#estim = LogisticRegression(solver='lbfgs' )
estim.fit(x_dftrain, y_dftrain)
Cs= estim.coef_


## Generate Whole Data set CM
df['calib_pred'] = df.apply(calibpred, axis=1)
df['calib_sex'] = df.apply(mof, axis = 1)
df['actual_class'] = df.actual_class.map({1:'Male',0:'Female'})
df['predicted_class'] = df.predicted_class.map({1:'Male',0:'Female'})
y_true = df.actual_class.tolist()
y_pred = df.calib_sex.tolist()
plot_confusion_matrix(y_true, y_pred, classes = ['Male','Female'], title = 'Whole Set')


## Generate Test Set CM
df_test['actual_class'] = df_test.actual_class.map({1:'Male',0:'Female'})
df_test['predicted_class'] = df_test.predicted_class.map({1:'Male',0:'Female'})
df_test['calib_pred'] = df.apply(calibpred, axis=1)
df_test['calib_sex'] = df.apply(mof, axis=1)
y_true = df_test.actual_class.tolist()
y_pred = df_test.calib_sex.tolist()
plot_confusion_matrix(y_true, y_pred, classes = ['Male','Female'], title = 'Test Set')

## While Loop
coef_1_list = []
coef_2_list = []
cm_array = []
TP_list = []
TN_list = []
FP_list = []
FN_list = []

i = 0
while i < 1000:
    calib_frac = .35
    df_train, df_test = train_test_split(df_while, test_size=calib_frac, stratify=df.actual_class)
    y_dftest = df_test.actual_class.values
    x_dftest = df_test.iloc[:, 5:7].values
    y_dftrain = df_train.actual_class.values
    x_dftrain = df_train.iloc[:, 5:7].values
    estim = LinearRegression(fit_intercept= False)
    estim.fit(x_dftrain, y_dftrain)
    coef_1_list.append(estim.coef_[0])
    coef_2_list.append(estim.coef_[1])
    Cs = estim.coef_
    df_test['actual_class'] = df_test.actual_class.map({1:'Male',0:'Female'})
    df_test['predicted_class'] = df_test.predicted_class.map({1:'Male',0:'Female'})
    df_test['calib_pred'] = df.apply(calibpred, axis=1)
    df_test['calib_sex'] = df.apply(mof, axis=1)
    y_true = df_test.actual_class.tolist()
    y_pred = df_test.calib_sex.tolist()
    cm_now = confusion_matrix(y_true, y_pred)
    cm_array.append(cm_now)
    TP_list.append(cm_now[0][0])
    TN_list.append(cm_now[1][1])
    FP_list.append(cm_now[1][0])
    FN_list.append(cm_now[0][1])
    i+=1

sum_of_cm = np.zeros([2,2])
for x in cm_array:
    grabbed_array = x
    sum_of_cm = grabbed_array + sum_of_cm

average_cm = sum_of_cm/len(cm_array)

normalized_average_cm = average_cm/(np.sum(average_cm))

cmfromcm(normalized_average_cm, classes = ['Male','Female'], title='1000 iterations, average confusion matrix, no average coeff', normalize = True)


Accuracy = (sum(TP_list) + sum(TN_list))/(sum(TP_list) + sum(TN_list) + sum(FP_list) + sum(FN_list))

























Cs[0] = np.mean(coef_1_list)
Cs[1] = np.mean(coef_2_list)

## Generate Whole Data predictions for n number of itterations 
df_while['calib_pred'] = df_while.apply(calibpred, axis=1)
df_while['calib_sex'] = df_while.apply(mof, axis = 1)
df_while['actual_class'] = df_while.actual_class.map({1:'Male',0:'Female'})
df_while['predicted_class'] = df_while.predicted_class.map({1:'Male',0:'Female'})
y_true = df_while.actual_class.tolist()
y_pred = df_while.calib_sex.tolist()
plot_confusion_matrix(y_true, y_pred, classes = ['Male','Female'], title = '25 iterations')



## Repeated Testing
Best_coeff = [-0.18073214, 0.59976251]



