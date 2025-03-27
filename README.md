# *Rooftop 3D Segmentation and Solar Potential Estimation* 
## 👋 Introduction
# Unet Structure
<img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" 
     alt="Unet" 
     width="500" />
---
# FCN Structure
<img src="https://discuss.pytorch.org/uploads/default/32008b38be5d436b1c0193c8aaa655d13d5ecda7" 
     alt="FCN" 
     width="600" />
---

This report aimes at predicting the segmentation of rooftop on a city to estimate the solar potential by using different models structures shown as above. The main idea to achieve the goal can be divided into three parts in general.

- Data Augmentation

     The images would be rotated from 90 degrees to 180, 270 degrees individually. And the images can also be flipped around x-axis.
  
- Model trainning

     The data would be splited into training set, validation set and testing set. Data-preprocessing would be needed, for instance, one of the most important ways is binarization where we set a threshold= 0.5, because the outputs of those model predicted were in the type of [0.] (float 32). All of these models (U-Net, FCN, and FCN_deep) are deep learning architectures implemented in TensorFlow. 
- Performance evaluation
     The metric IOU would be used to evaluate the model representing the model gerneralization ability.
   
Download the dataset: 
Data location: https://dataserv.ub.tum.de/index.php/s/m1655470
Create a local folder on your desktop and clearly lable locations of you images (training data) their labels (masks) and testing data
