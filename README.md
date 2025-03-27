# *Rooftop 3D Segmentation and Solar Potential Estimation* 
## 👋 Introduction
# Unet Structure
<img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" 
     alt="Unet" 
     width="500" />
---
# FCN & FCN_deep Structure
<img src="https://discuss.pytorch.org/uploads/default/32008b38be5d436b1c0193c8aaa655d13d5ecda7" 
     alt="FCN" 
     width="600" />
---

This report aimes at predicting the segmentation of rooftop on a city to estimate the solar potential by using different models structures shown as above. The main idea to achieve the goal can be divided into three parts in general.

- Data Augmentation

     The images would be rotated from 90 degrees to 180, 270 degrees individually shown in geometric_augmentatiom. And the images can also be flipped around x-axis shown in the function of x_axis_reflection . If you have a large GPU you can try geometric_augmentatiom.
  
- Model trainning

     The data are split into training, validation, and testing sets. Data preprocessing is required; for example, one important step is binarization by setting a threshold of 0.5, because the model outputs are floating-point predictions in the range [0, 1]. All of these models (U-Net, FCN, and FCN_deep) are deep learning architectures implemented in TensorFlow.
   
- Performance evaluation
  
     The IoU metric (Intersection over Union) is used to evaluate the performance of the segmentation model, reflecting how well it generalizes to unseen data.
  
---
<a id="license"></a>
## License
data:https://dataserv.ub.tum.de/index.php/s/m1655470. 

---

Create a local folder on your desktop or pc and clearly lable locations of you images (training data) their labels (masks) and testing data.

---

## Requirements: 
cuda == 11.2
TensorFlow == 2.10

---

## Libraries:(Below are some of the main libraries that are used)
- os
- tensorflow
- matplotlib
