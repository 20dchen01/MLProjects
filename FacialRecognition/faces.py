import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import sklearn.model_selection
from sklearn.cross_validation import train_test_split
import pathlib
train_csv = pd.read_csv('dataplant/train.csv')# Prepend image filenames in train/ with relative path
filenames = ['train/' + fname for fname in train_csv['id'].tolist()]
labels = train_csv['has_cactus'].tolist()
train_filenames, val_filenames, train_labels, val_labels = train_test_split(filenames,labels,train_size=0.9,random_state=42)

train_data = tf.data.Dataset.from_tensor_slices(
  (tf.constant(train_filenames), tf.constant(train_labels))
)
val_data = tf.data.Dataset.from_tensor_slices(
  (tf.constant(val_filenames), tf.constant(val_labels))
)
IMAGE_SIZE = 96 # Minimum image size for use with MobileNetV2
BATCH_SIZE = 32# Function to load and preprocess each image
def _parse_fn(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = (tf.cast(img, tf.float32)/127.5) - 1
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, label
# Run _parse_fn over each example in train and val datasets
# Also shuffle and create batches
train_data = (train_data.map(_parse_fn)
             .shuffle(buffer_size=10000)
             .batch(BATCH_SIZE)
             )
val_data = (val_data.map(_parse_fn).shuffle(buffer_size=10000).batch(BATCH_SIZE))