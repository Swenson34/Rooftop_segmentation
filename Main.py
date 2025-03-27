# Import required packages
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from sklearn.model_selection import train_test_split


# Import made functions
from Models import Unet, FCN, FCN_deep
from Performance_metric_IoU import IoU
from Training import model_training
from Data_augmentation import geometric_augmentation, x_axis_reflection

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

# Create masks with excluded segments which are our target variable

# Empty tensor
Y_seg_full = np.zeros((len(mask_ids[2]), img_height, img_width,1), dtype = np.bool_)

# Populate the tensor
for n, id in tqdm(enumerate(train_ids[2]), total=len(train_ids[2])):
    mask = imread(path_mask + "\\" + id.strip('.tif') + ".png")
    mask_seg = imread(path_mask_seg + "\\" + id.strip('.tif') + ".png")

    mask = (mask != 17)  # Label rooftop as True
    mask_seg = (mask_seg != 8)  # Label superstructures as True

    mask = tf.reshape(mask, [img_height, img_width, 1])
    mask_seg = tf.reshape(mask_seg, [img_height, img_width, 1])

    Y_seg_full[n] = ((mask == True) & (mask_seg == False))

# Split the data into training, validation and test set
# Having the same seed as in the first split should ensure the same indices are used

Y_train_with_val_seg, Y_test_seg = train_test_split(Y_seg_full, test_size=0.1, random_state=52, shuffle=True)
Y_train_seg, Y_val_seg = train_test_split(Y_train_with_val_seg, test_size=1/9, random_state=52, shuffle=True)

# Augment the data set
# If running the code on large GPU, replace x_axis_reflection with geometric_augmentation

# Training data including validation set
X_augmented = x_axis_reflection(X_train_with_val,  img_height, img_width, img_channels, np.uint8)
Y_seg_augmented = x_axis_reflection(Y_train_with_val, img_height, img_width, 1, np.bool_)

# Training data without validation set
X_train_augmented = x_axis_reflection(X_train,  img_height, img_width, img_channels, np.uint8)
Y_train_seg_augmented = x_axis_reflection(Y_train_seg, img_height, img_width, 1, np.bool_)


## The code below has been commented out as it running all of it would be computational expensive.
## Instead, uncomment the individual parts that you require and run the code.
## Once models have been trained, their predictions can be visualised by running Performance_visualisation.py


# If models have already been trained and saved weights are available skip the next step
# Alternatively, uncomment the models you want to train

# Unet_model = model_training(Unet(img_width, img_height, img_channels), x_train = , y_train = , x_valid = , y_valid = , path_to_save_model = , path_to_logs  = , batch_size_ = 8, epochs_ = 25)
# FCN_model = model_training(FCN(img_width, img_height, img_channels), x_train = , y_train = , x_valid = , y_valid = , path_to_save_model = , path_to_logs = , batch_size_ = 8, epochs_ = 25)
# FCN_deep_model = model_training(FCN_deep(img_width, img_height, img_channels), x_train = , y_train = , x_valid = , y_valid = , path_to_save_model = , path_to_logs = , batch_size_ = 8, epochs_ = 25)

# Load models

# Unet_model = keras.models.load_model("your_path")
# FCN_model = keras.models.load_model("your_path")
# FCN_deep_model = keras.models.load_model("your_path")


# Define a threshold for segmentation
# This will include some trial and error on validation data to find the optimal value

# threshold = 0.5
#
# Unet_model_pred = Unet_model.predict(X_val, verbose=1)
# Unet_model_pred = (Unet_model_pred > threshold).astype(np.uint8)
#
# FCN_model_preds = FCN_model.predict(X_val, verbose=1)
# FCN_model_preds = (FCN_model_preds > threshold).astype(np.uint8)
#
# FCN_deep_model_preds = FCN_deep_model.predict(X_val, verbose=1)
# FCN_deep_model_preds = (FCN_deep_model_preds > threshold).astype(np.uint8)

# Evaluate the model

# Model_performance = IoU( Target = , Prediction = )
# print(Model_performance)
