# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:05:05 2018

@author: david
"""

import os
import h5py
import numpy as np


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
from sklearn.externals import joblib

dirpath = os.getcwd()
print("Current directory is : " + dirpath)
plantname=input("input plant name: ")
test_samples = os.path.join("output/", plantname, (plantname + "_test_samples6.h5"))
test_sample_features = os.path.join("output/", plantname, (plantname + "_test_features6.h5"))
trainedmodelfile=os.path.join("output/", plantname, (plantname + "_train_model"))
trainedscaler= os.path.join("output/", plantname, (plantname + "_train_scaler"))

#featuregenerationresultfile="output/Cherry_data.h5"
#train_features_results="output/Corn_train_features6.h5"
#test_sampling_results="output/Corn_test_samples6.h5"
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

scaler=joblib.load(trainedscaler)
clf=joblib.load(trainedmodelfile)

with h5py.File(test_samples,'r') as g:
    testimages_string = g['Samples']
    testimages = np.array(testimages_string)
g.close()

# import the test file names
with h5py.File(test_sample_features,'r') as g:
    global_features_string = g['Samples']
    global_labels_string = g['Labels']
    test_features = np.array(global_features_string)
    test_labels = np.array(global_labels_string)
    ClassNames_string = g['ClassNames']
    ClassNames = np.array(ClassNames_string)
g.close()
print("test samples: ",len(test_labels))
#generate test sample features
#print("generate test features...")
#(test_global_features,test_global_labels,ClassNames,Sample_path)=featuregen.feature_generate(test_sampling_results)

print("ClassNames", ClassNames)

# normalize the feature vector using train scaler result
# the scaler was trainned in train section, now use the train result to normalize test set--HC
test_global_features = scaler.transform(np.array(test_features)) #should use.transform--HC

print ("Test data   : {}".format(test_global_features.shape))
print ("Test labels : {}".format(np.array(test_labels).shape))


prediction = clf.predict(test_global_features)

#print("test sample label: ",testLabelsGlobal)
#print("predictions of test sample label: ",prediction)
devirationofpredictions = prediction -  test_labels
correctpredictions = 0

print("error predictions: ")
for x in range(0, len(test_labels)):    
    if devirationofpredictions[x]>0:
        print("error for: ", testimages[x])
        print("test sample type",ClassNames[test_labels[x]],"prediction type", ClassNames[prediction[x]])
    elif devirationofpredictions[x]<0:
        print("error for: ", testimages[x])
        print("test sample type",ClassNames[test_labels[x]],"prediction type", ClassNames[prediction[x]])
    elif devirationofpredictions[x]==0:
        correctpredictions=correctpredictions+1


print("total test sample: ", len(test_labels))
print("correctpredictions predictions are: ", correctpredictions)
labelclasses=np.arange(len(ClassNames))
print("ClassNames", ClassNames)
print("labelclasses: ",labelclasses)

testsamplenumber=np.arange(len(labelclasses))
Real_Positive=np.arange(len(labelclasses),dtype='float')
Real_Negative=np.arange(len(labelclasses),dtype='float')
False_Positive=np.arange(len(labelclasses),dtype='float')
False_Negative=np.arange(len(labelclasses),dtype='float')

for i in labelclasses:   
    testsamplenumber[i]=np.sum(test_labels==i)
    pds=prediction[test_labels==i]
    nds=prediction[test_labels!=i]
    Real_Positive[i]=np.sum(pds[pds==i]==i)/np.sum(prediction[prediction==i]==i) #class i and is predicted as class i
    Real_Negative[i]=np.sum(nds[nds!=i]!=i)/np.sum(prediction[prediction!=i]!=i)#not class i and is predicted as not class i
    False_Positive[i]=np.sum(nds[nds==i]==i)/np.sum(prediction[prediction==i]==i) #not class i but is predicted as class i
    False_Negative[i]=np.sum(pds[pds!=i]!=i)/np.sum(prediction[prediction!=i]!=i)   
print("totalsample: ",testsamplenumber)
print("Real_Positive: ", Real_Positive)
print("Real_Negative: ", Real_Negative)
print("False_Positive: ", False_Positive)
print("False_Negative: ", False_Negative)
#print("test labels: ",testLabelsGlobal)
#print("predictions labels: ",prediction)

types=len(labelclasses)
rs=np.zeros(shape=(types,types),dtype='int')
for i in labelclasses: 
    for j in labelclasses:
        rs[i][j]=np.sum(prediction[test_labels==i]==j)
print("result metrix, row is real type, column is prediction type:")
print(rs)

print ("clf scores: ", clf.score(test_global_features, test_labels))

