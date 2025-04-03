
![Repository Avatar](./ML_denoising_logo.webp)

# DASDenoisingML_MultipleInterrogators
Machine Learning (ML) tools for removing instrumental noise in Distributed Acoustic Sensing (DAS) data.

# Overview
This folder contains Jupyter notebooks and Python utilities for removing instrumental noise in DAS data acquired with three different interrogators. For more information on the training and application of the ML models, please refer to Gu et al. (2024).

## Jupyter Notebooks
In the repository, there are several example notebooks that can be run on field data.

## Data
The *Data* folder contains the data used in the notebooks. These data examples consist of DAS Vertical Seismic Profile data acquired at the Curtin National Geosequestration Laboratory, Perth, Australia. The VSP data were acquired using a 45 kg accelerated weight drop source. The downhole DAS cable was connected to three different interrogator units, as specified by the subfolder names. 

## Models
The *Models* folder contains pre-trained denoising neural networks for each type of interrogator.

## Utils
The *Utils* folder contains processing utilities used in the Jupyter notebooks.

## Reference
Gu, X., Collet, O., Tertyshnikov, K., & Pevzner, R. (2024). Removing Instrumental Noise in Distributed Acoustic Sensing Data: A Comparison Between Two Deep Learning Approaches. *Remote Sensing, 16(22)*, 4150.
