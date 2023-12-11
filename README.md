<h1 align="center":> Convolutional Neural Networks (CNNs) to Identify Malignant Moles </h1>

<p align="center">
  <img src="/readme_images/mole.jpeg" alt="mole" width="50%" />
</p>

## Table of Contents 
1. [➤ About The Project](#About-the-Project)
2. [➤ Dataset](#Dataset)
3. [➤ Project Files](#Project-Files)


## About the Project 
This project investigates several different convolutional neural network (CNN) architectures 
and adapts them to the task of malignant (cancerous) mole identification. Melanoma skin cancers have a 5-year survival rate of nearly 99% when detected and treated early. Around 30\% of existing moles arise from malignant moles. Though these malignant moles can be identified and removed by dermatologists it is generally difficult for an untrained professional to distinguish a malginant mole from a benign mole. Thus, early detection of malignant moles often relies on regular visits to a dermatologist. 

This project aims to adapt CNNs to the task of identifying malignant moles. The trained CNNs may improve the ability of people to detect malignant moles early by providing them with a method of identifying malignant moles from home. 

## Dataset 
The dataset used for this project is publicly-available from 
[Kaggle](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images). The dataset consists of 10,605 dermoscopy images. 5,105 images are of malignant moles and the remaining 5,500 images are of benign moles. 

## Project Files
1. `Project Report.pdf` || contains a report of the research performed on this project.
2. `src/` || constains python modules for creating visualizations, model training, data transformations and data loading.
3. `VGG11_Model_Training.ipynb` || notebook for training the VGG11 model.
4. `VGG11_Model_Teting.ipynb` || notebook for evaluating the performance of the VGG11 model.
5. `AlexNet_Model_Training.ipynb` || notebook for training the AlexNet model.
6. `AlexNet_Model_Testing.ipynb` || notebook for testing the AlexNet model.
7. `Tiny_VGG_Model_Training.ipynb` || notebook for training the tiny vgg model.
8. `Tiny_VGG_Model_Testing.ipynb` || notebook for testing the tiny vgg model.

