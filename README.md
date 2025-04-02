# *Rooftop 3D Segmentation and Solar Potential Estimation* 
# Introduction
This repository contains the entirety of our work for EE981 Image and Video Processing module project. The tile of our project was "Rooftop 3D Segmentation and Solar Potential Estimation" and its objectives were the following: 

     1. Identify the benefits and disadvantages of using solar panels to the environment.  
     
     2. Conduct a comparison of the two approaches: LiDAR and CV for evaluation of the PV potential.  
     
     3. Carry out a literature review of CV-based approaches for rooftop segmentation and evaluation of PV potential.  
     
     4. Analyse the pipeline and identify the limitations of the study, “RID—Roof Information Dataset for Computer Vision-Based Photovoltaic Potential Assessment.”  
     
     5. Map any literature findings that could improve the process of said study and mitigate some of the limitations.  
     
     6. Implement different Neural Network architectures to perform analysis of rooftop data and evaluate the accuracy of each, using the study’s existing dataset.    
     
     7. Estimate PV potential using the results from each neural network.  

# Methodology
The first step of our methodology is data preparation. We used data set from: “RID—Roof Information Dataset for Computer Vision-Based Photovoltaic Potential Assessment” which contains 1880 aerial images of centred buildings from a village of Wartenberg, Germany. Along with aerial images of buildings, the paper provides two annotated mask data sets. The first one contains labels of the rooftop area (without excluding segments), while the second one contains labels of the rooftop segments. Using these two masks, the third mask was created. It contains only the rooftop area without any superstructures and is the dependent variable of the study.

To predict the dependent variable (rooftop area without any superstructures), three Neural Network models were defined. These are: U-net with around 2 million parameters, Fully Convolutional Network (FCN) with also around 2 million parameters and FCN revised (deep) model with around 11 million parameters. The defined models can be viewed and edioted in the Models.py file.

Once the models were trained, we evaluated them on validation data using using Intersection over Union (IoU) metric, which can be viewed in Performance_metric_IoU.py file.

The predictions of the models are used to calculate the area of rooftops without any superstructures. This is done by counting the number of pixels in predicted output and multiplying them by the area each pixels captures. For our images, this was 0.01 metere squared.

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
# Model performance
# U-net without augmentation out of sample:
![Image](https://github.com/user-attachments/assets/507201fe-0c3a-4b64-b91f-d4683df1fbdd)

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
