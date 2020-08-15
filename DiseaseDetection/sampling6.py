# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:15:13 2018

@author: david
"""

from sklearn.model_selection import train_test_split
import numpy as np
import os
import featuregen
#import featuregen #module of feature generation, build by--HC



#Input data path, result data file's name and path-HC
#--------------------------------------------------------
#for program development:
#cherry:
#sample_path = "dataset/Cherry"
#sampling_results = "output/Cherry_sampling.h5";
#Corn
sample_path = "dataset/Corn"
sampling_train_results = "output/Corn_train_samples6.h5"
sampling_test_results = "output/Corn_test_samples6.h5"
#------------------------------------------------------

dirpath = os.getcwd()
print("Current directory is : " + dirpath)
#sample_path=input("Input sample dataset path:")
#sampling_results=input("Input sampling result datafile and path:")
testfraction=input("Input test_size (0-1): ")
#testfraction=0.1


#Data sample labels generation: -HC
#Using the dataset fold name as labels: -HC
sample_labels = os.listdir(sample_path)
SNlabels=np.arange(len(sample_labels)) #numeric labels, =[0,1,2,3,...]

#initlization--HC
# empty lists to hold feature vectors, labels and image file names
labels = []
imagesamples=[]
global_features = []
TrainSamples=[]
TestSamples=[]
TrainLabels=np.array([],dtype=int)
TestLabels=np.array([],dtype=int)
j=0#imagefiles =[]

# feature generation: --HC
# loop over the training data sub-folders
# loop over the training data sub-folders
for i in SNlabels:
    # join the training data path and each species training folder
    dir = os.path.join(sample_path, sample_labels[i])
    print ("STATUS: processing data set: ", sample_labels[i])#--HC
    # get the current training label
    current_label = i
    sample_files = os.listdir(dir)
    images_per_class = len(sample_files)
    print("image number:",images_per_class)
    # loop over the images in each sub-folder
    for x in range(1, images_per_class+1):
        # get the image file name
        image_file = dir + "/" + sample_files[x-1]
        # update the list of labels and feature vectors
        labels.append(current_label)
        imagesamples.append(image_file) #create image file name including subdir name for tracking later-HC
        j += 1
    print ("[STATUS]: ", x, "images in folder: {}".format(sample_labels[i]), "processed.")
print ("[STATUS] completed Global Feature Extraction...")
(TrainSamples, TestSamples, TrainLabels, TestLabels) = train_test_split(np.array(imagesamples),
     np.array(labels),test_size=float(testfraction), random_state=9)


print ("[STATUS] completed sampling...")


# save the sampling train result in a HDF5 file: 
featuregen.data_save(sampling_train_results,TrainSamples,TrainLabels,sample_labels,sample_path)
featuregen.data_save(sampling_test_results,TestSamples,TestLabels,sample_labels,sample_path)

#show results
print ("sample image data are from: ",sample_path)
print ("train sample file is: ",sampling_train_results) 
print ("test sample file is: ",sampling_test_results) 
print ("total sample number: ", j)    
print ("total train sample number: ", len(TrainSamples))
print ("total test sample number: ",len(TestSamples))
print ("[STATUS] Random sampling, test size: ", testfraction)
print ("[STATUS] end of sampling")




