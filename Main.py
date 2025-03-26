# Import required packages
import tensorflow as tf
import keras
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
import random
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import collections
import multiprocessing
import h5py
from sklearn.model_selection import train_test_split


# Import made functions
from Models import Unet, FCN, FCN_deep_model
from Performance_metric_IoU import IoU
from Training import model_training

# Size of inputs
img_width = 512
img_height = 512
img_channels = 3

# Path to data
train_path = "C:\\Users\\Sven\\Desktop\\EE981project\\Data\\images_roof_centered_geotiff"
path_mask = "C:\\Users\Sven\\Desktop\\EE981project\Data\\masks_segments_reviewed"
path_mask_seg = "C:\\Users\\Sven\Desktop\\EE981project\\Data\masks_superstructures_reviewed"

train_ids = next(os.walk(train_path))
mask_ids = next(os.walk(path_mask))
mask_seg_ids = next(os.walk(path_mask_seg))

# Empty tensor that will be populated by data
X_full = np.zeros((len(train_ids[2]), img_height, img_width, img_channels), dtype = np.uint8)
Y_full = np.zeros((len(mask_ids[2]), img_height, img_width,1), dtype = np.bool_)

# Populate the matrix
for n, id in tqdm(enumerate(train_ids[2]), total = len(train_ids[2])):
    path = train_path + "\\" + id
    img = imread(path)[:,:,:img_channels]
    X_full[n] = img
    mask = imread(path_mask + "\\" + id.strip('.tif')+".png")
    mask = (mask != 17) # 17 is background
    Y_full[n] = tf.reshape(mask, [img_height, img_width,1])

# X_full and Y_full contain the whole data set

# Split the data into training, validation and test set using 80/10/10 split

X_train_with_val, X_test, Y_train_with_val, Y_test = train_test_split(X_full, Y_full, test_size=0.10, random_state=52, shuffle=True)

X_train, X_val, Y_train, Y_val = train_test_split(X_train_with_val, Y_train_with_val, test_size=1/9, random_state=52, shuffle=True)

