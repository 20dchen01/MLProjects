# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:05:05 2018

@author: david
"""

import os
import h5py
import numpy as np
#import featuregen
#import cv2
#import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


dirpath = os.getcwd()
print("Current directory is : " + dirpath)

#featuregenerationresultfile=input("Input featuregenerationresultfile's path and name:")

#featuregenerationresultfile="output/Cherry_data.h5"
train_features_results="output/Corn_train_features6.h5"
demo_path="dataset/corn/demo"
trainedmodelfile="output/Corn_train_model.h5"
trainedscaler="output/Corn_train_scaler.h5"

# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=9)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=9)))

# variables to hold the results and names
results = []
names = []
scoring = "accuracy"

# import the training feature vector and labels
with h5py.File(train_features_results,'r') as g:
    global_features_string = g['Samples']
    global_labels_string = g['Labels']
    train_features = np.array(global_features_string)
    train_labels = np.array(global_labels_string)
    ClassNames_string = g['ClassNames']
    ClassNames = np.array(ClassNames_string)
g.close()
print("ClassNames", ClassNames)

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
train_features = scaler.fit_transform(train_features)
joblib.dump(scaler,trainedscaler)

print ("Train data  : {}".format(train_features.shape))
print ("Train labels: {}".format(train_labels.shape))



clf  = RandomForestClassifier(n_estimators=100, random_state=9)
#clf=KNeighborsClassifier(n_neighbors=1,weights='distance')
# fit the training data to the model
clf.fit(train_features, train_labels)
joblib.dump(clf,trainedmodelfile)

print("Trained scaler saved in: ",trainedscaler)
print("Trained classifier saved in: ",trainedmodelfile)
print("end of the train process")



