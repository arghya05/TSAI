# Submission for Week 8 (Late Assignment ON Time)

- [Problem Statement](#problem-statement)
- [Results & Analysis](#results-analysis)
- [CIFAR-10 Augmentation Vizulation](#cifar-10-augmentation-vizualization)
- [Model Evaluation](#model-evaluation)
  * [ResNet 18 Learning Curve](#resnet-18-learning-curve)
  * [ResNet 18 Misclassified Images](#resnet-18-misclassified-images)
  * [Grad-Cam ResNet 18](#gradcam-resnet18)
- [Team Members](#team-members)


# Problem Statement

1. Write a custom ResNet architecture for CIFAR10 that has the following architecture:

   1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
   2. Layer1 -
      1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
      2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
      3. Add(X, R1)
   3. Layer 2 -
      1. Conv 3x3 [256k]
      2. MaxPooling2D
      3. BN
      4. ReLU
   4. Layer 3 -
      1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
      2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
      3. Add(X, R2)
   5. MaxPooling with Kernel Size 4
   6. FC Layer 
   7. SoftMax
2. Uses One Cycle Policy such that:
   1. Total Epochs = 24
   2. Max at Epoch = 5
   3. LRMIN = FIND
   4. LRMAX = FIND
   5. NO Annihilation
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Target Accuracy: 90% **(93% for late submission or double scores)**. 

# Results Analysis
Link to [Notebook](https://github.com/vivek-a81/EVA6/blob/main/Session8/session_8.ipynb)

Link to [Main Repo](https://github.com/MittalNeha/vision_pytorch)
- Test Accuracy : 88.01%
- Train Accuracy : 87.74%
- In the last layer of ResNet18 we have used stide of 1
- We also trained ResNet 34 which overfitted on class **Truck**



# CIFAR-10 Augmentation Vizualization

- **DataSet:** CIFAR-10 has **10 classes** of **32,32** that are **Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship & Truck**

<p float="center">
  <img src="images/aug.png" alt="drawing" width="650" height="550">
</p>


# Model Evaluation



ResNet 18 Misclassified Images
--------------------------




GradCam ResNet18
--------------------------



References
------------------------

* https://www.kaggle.com/gilf641/lr-finder-using-pytorch
* http://gradcam.cloudcv.org


Team Members
------------------------

Neha Mittal, Vivek Chaudhary

