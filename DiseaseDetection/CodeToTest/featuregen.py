# Feature generation from image
"""
Created on Fri Aug 17 10:15:13 2018

@author: david
"""

import numpy as np
import mahotas
import cv2
import h5py


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hu_moments = cv2.HuMoments(cv2.moments(image)).flatten()
    return hu_moments

def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

def fd_histogram(image, bins):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

def feature(image_file,fixed_size,bins):
    image = cv2.imread(image_file)
    image = cv2.resize(image, fixed_size)
    ####################################
    # Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image,bins)

    ###################################
    # generate feature metrix
    ###################################
    feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    return feature

def feature_generate (samples):
    # fixed-sizes for image
    fixed_size = tuple((500, 500))
    # bins for histogram
    bins = 8

    # import training sampling results:--HC
    with h5py.File(samples,'r') as g:
        Samples_string = g['Samples']
        Labels_string = g['Labels'] #this numeric labels
        ClassNames_string = g['ClassNames']
        Samples = np.array(Samples_string)
        Labels = np.array(Labels_string)
        ClassNames = np.array(ClassNames_string)
        Sample_path_string= g['SamplePath']
        Sample_path=np.array(Sample_path_string)[0]
        g.close()
    totalsample=len(Samples)
    global_features = []
    # feature generation: --HC
    # loop over the training data sub-folders
    print("Processing and feature generation...")
    for x in range(1,totalsample+1):
        image_file=Samples[x-1]
        image_feature = feature(image_file,fixed_size,bins)
        global_features.append(image_feature)
    labels=Labels.tolist() #convert to list for counting img file number in each type--HC
    for i in range (1,len(ClassNames)+1):
        print("Image numbers of ", ClassNames[i-1], " data set: ",labels.count(i-1))  
    print("[STATUS] completed Global Feature Extraction...")
    return (global_features,labels,ClassNames,Sample_path)   


def data_save(filename,samples,labels,sample_labels,sample_path):
    dt = h5py.special_dtype(vlen=str) #define data type of str for dataset creating later-HC
    with h5py.File(filename, 'w') as f:
        f.create_dataset('Labels', data=np.array(labels))
 #       samplesource=np.array(datasource)
        Samples=np.array(samples)  #converting list to array, as list has no shape properity-HC
        Class_name=np.array(sample_labels)  
        imfs = f.create_dataset('Samples', Samples.shape,dtype=dt)
        classname = f.create_dataset('ClassNames', Class_name.shape,dtype=dt)
        datasource=f.create_dataset('SamplePath', (1,),dtype=dt) 
        imfs[:] = Samples
        classname[:]=Class_name
        datasource[0]=sample_path
    f.close()
    return data_save
