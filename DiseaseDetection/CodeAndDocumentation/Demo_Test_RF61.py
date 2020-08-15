# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:05:05 2018

@author: david
"""

import os
import h5py
import numpy as np
import featuregen
import cv2
import matplotlib.pyplot as plt
from sklearn.externals import joblib
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC


dirpath = os.getcwd()
print("Current directory is : " + dirpath)
plantname=input("input plant name: ")
demo_path = os.path.join("dataset/demo/", plantname)
train_features_results = os.path.join("output/", plantname, (plantname + "_train_features6.h5"))
trainedmodelfile=os.path.join("output/", plantname, (plantname + "_train_model"))
trainedscaler= os.path.join("output/", plantname, (plantname + "_train_scaler"))
#featuregenerationresultfile=input("Input featuregenerationresultfile's path and name:")

#featuregenerationresultfile="output/Cherry_data.h5"
#train_features_results="output/Corn_train_features6.h5"
#demo_path="dataset/corn_demo"
#trainedmodelfile="output/Corn_train_model.h5"
#trainedscaler="output/Corn_train_scaler.h5"

# create all the machine learning models
#models = []
#models.append(('LR', LogisticRegression(random_state=9)))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier(random_state=9)))
#models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=9)))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(random_state=9)))

# variables to hold the results and names
results = []
names = []
scoring = "accuracy"

# import the training feature vector and labels
with h5py.File(train_features_results,'r') as g:
#    global_features_string = g['Samples']
#    global_labels_string = g['Labels']
#    train_features = np.array(global_features_string)
#    train_labels = np.array(global_labels_string)
    ClassNames_string = g['ClassNames']
    ClassNames = np.array(ClassNames_string)
g.close()
#print("ClassNames", ClassNames)

# normalize the feature vector in the range (0-1)
#scaler = MinMaxScaler(feature_range=(0, 1))
#train_features = scaler.fit_transform(train_features)
scaler=joblib.load(trainedscaler)

#print ("Train data  : {}".format(train_features.shape))

#print ("Train labels: {}".format(train_labels.shape))



#clf  = RandomForestClassifier(n_estimators=100, random_state=9)
#clf=KNeighborsClassifier(n_neighbors=1,weights='distance')
# fit the training data to the model
#clf.fit(train_features, train_labels)
clf=joblib.load(trainedmodelfile)

# import the test file names
# fixed-sizes for image
fixed_size = tuple((500, 500))
# bins for histogram
bins = 8
print("demo_path",demo_path)
demofiles = os.listdir(demo_path)
for filename in demofiles:
     # read the image
    image_file = os.path.join(demo_path, filename)
    image = cv2.imread(image_file)
    # resize the image
    image = cv2.resize(image, fixed_size)
    image_feature = featuregen.feature(image_file,fixed_size,bins)
    # normalize the feature vector using train scaler result
    # the scaler was trainned in train section, now use the train result to normalize test set--HC
    test_global_feature = scaler.transform(np.array(image_feature).reshape(1,-1)) #should use.transform--HC
    prediction = clf.predict(test_global_feature.reshape(1,-1))[0]
    

   #show predicted label on image
    cv2.putText(image, ClassNames[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    print("file name:",filename)
    print("prediction lable:", ClassNames[prediction], "\n")




