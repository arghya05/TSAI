# Submission for Week 7


# Problem Statement

- Fix the network above:
  - change the code such that it uses GPU
  - change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
  - total RF must be more than 52
  - two of the layers must use Depthwise Separable Convolution
  - one of the layers must use Dilated Convolution
  - use GAP (compulsory mapped to # of classes):- CANNOT add FC after GAP to target #of classes 
  - use albumentation library and apply:
  - horizontal flip
  - shiftScaleRotate 
  - coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
  - grayscale
  - achieve 87% accuracy, as many epochs as you want. Total Params to be less than 100k


# CIFAR-10 Vizualization & Augmentation

- DataSet

<p float="center">
  <img src="images/visualization_2.png" alt="drawing" width="700" height="550">
</p>

- Agumentation 

<p float="center">
  <img src="images/visualization_1.png" alt="drawing" width="700" height="550">
</p>



# Out Model
