# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:15:13 2018

@author: david
"""

#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import featuregen #module of feature generation, build by--HC


        
#Input data path, result data file's name and path-HC
#--------------------------------------------------------
#for program development:
#cherry:
#train_path = "dataset/Cherry"
#featuregenerationresultfile = "output/Cherry_data.h5";
#Corn
#train_path = "dataset/Corn"
#featuregenerationresultfile = "output/Corn_data.h5"
#------------------------------------------------------

dirpath = os.getcwd()
print("Current directory is : " + dirpath)
#sampling_results=input("Input sampling file path:")
#featuregenerationresultfile=input("Input featuregenerationresultfile's path and name:")
#sampling_results="output/Cherry_sampling.h5"
#featuregenerationresultfile="output/Cherry_data.h5"

train_sampling_results="output/Corn_train_samples6.h5"
#test_sampling_results="output/Corn_test_samples.h5"
train_features_results="output/Corn_train_features6.h5"
#test_features_results="output/Corn_test_features6.h5"

print("processing train data....")
(train_global_features,train_labels,ClassNames,Sample_path)=featuregen.feature_generate(train_sampling_results)
featuregen.data_save(train_features_results,train_global_features,train_labels,ClassNames,Sample_path)

#print("processing test data....")
#(test_global_features,test_labels,ClassNames,Sample_path)=featuregen.feature_generate(test_sampling_results)
#featuregen.data_save(test_features_results,test_global_features,test_labels,ClassNames,Sample_path)


#show feature vector and label size: --HC
# get the overall feature vector size
print ("[STATUS] train feature vector size {}".format(np.array(train_global_features).shape))
# get the overall training label size
print ("[STATUS] train Labels {}".format(np.array(train_labels).shape))
# get the overall feature vector size
#print ("[STATUS] test feature vector size {}".format(np.array(test_global_features).shape))
# get the overall training label size
#print ("[STATUS] test Labels {}".format(np.array(test_labels).shape))



#show results
print ("sampling image data are from: ",Sample_path)
print ("feature generation results of train data file is: ",train_features_results)   
#print ("feature generation results of test data file is: ",test_features_results)    
print ("[STATUS] end of feature generation")




