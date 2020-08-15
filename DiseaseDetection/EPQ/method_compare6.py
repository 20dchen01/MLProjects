# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:05:05 2018

@author: david
"""

import os
import h5py
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


dirpath = os.getcwd()
print("Current directory is : " + dirpath)
plantname=input("input plant name: ")
featuregenerationresultfile  = os.path.join("output/", plantname, (plantname + "_train_features6.h5"))


#Cherry
#featuregenerationresultfile = "output/Cherry_data.h5";
#featuregenerationresultfile = "output/Corn_train_features6.h5";


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

# import the feature vector and trained labels
with h5py.File(featuregenerationresultfile,'r') as g:
    global_features_string = g['Samples']
    global_labels_string = g['Labels']
    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)
    ClassNames_string = g['ClassNames']
    ClassNames = np.array(ClassNames_string)
 #   TargerNames=np.array(tgns)
g.close()
print("classnames", ClassNames)

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
global_features = scaler.fit_transform(global_features)

print ("[STATUS] splitted train and test data...")

#(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),np.array(global_labels), test_size=0.0, random_state=9)


#print ("[STATUS] splitted train and test data...")

print ("Train data  : {}".format(global_features.shape))
#print ("Test data   : {}".format(testDataGlobal.shape))
print ("Train labels: {}".format(global_labels.shape))
#print ("Test labels : {}".format(testLabelsGlobal.shape))

# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, global_features, global_labels, cv=kfold, scoring=scoring)
    print (cv_results)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()




