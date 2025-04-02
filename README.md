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
# Model performance visualised
In the following we show an example of each of the built models predictions out-of-sample on test data. The top image is the function input; bottom left is the ground truth; and bottom right is the models' prediction.
# U-net without augmentation out of sample:
![Image](https://github.com/user-attachments/assets/48fbdcca-b6f9-4e97-bf07-ba39ee3c5351)

# U-net with augmentation out of sample:
![Image](https://github.com/user-attachments/assets/64c34a64-a004-4acf-87f0-a93a137d1ecf)

# FCN without augmentation out of sample:
![Image](https://github.com/user-attachments/assets/2dfa7b8d-b4eb-483d-9666-44cdef31c6fc)

# FCN with augmentation out of sample:
![Image](https://github.com/user-attachments/assets/b26f2f0d-a122-4cb5-af20-c80d5e710706)

# FCN revised without augmentation out of sample:
![Image](https://github.com/user-attachments/assets/a19ea7a1-695e-4e59-a28b-43baf4625469)

# FCN revised with augmentation out of sample:
![Image](https://github.com/user-attachments/assets/70c5f71c-6998-4638-a0b5-a052cf3f2052)
---
# Model performance IoU
![Image](https://github.com/user-attachments/assets/7f25182b-f7d5-4653-9da3-2943f8fa2ef8)


    Run	U-Net	FCN       FCN Revised
     1	0.8695	0.8673	0.8274
     2	0.8658	0.8670	0.8228
     3	0.8757	0.8786	0.8516
     4	0.8646	0.8649	0.8668

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
