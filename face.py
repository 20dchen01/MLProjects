# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:21:00 2020

@author: david
"""


import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

seed = 21
datagen = ImageDataGenerator()
train_it = datagen.flow_from_directory('data/train/', class_mode='binary', batch_size=30)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('data/validation/', class_mode='binary', batch_size=30)
# load and iterate test dataset
test_it = datagen.flow_from_directory('data/test/', class_mode='binary', batch_size=30)

model = ...
# fit model
model.fit_generator(train_it, steps_per_epoch=4, validation_data=val_it, validation_steps=8)
loss = model.evaluate_generator(test_it, steps=24)
yhat = model.predict_generator(test_it, steps=24)
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))