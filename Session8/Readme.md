# Submission for Week 7 (Late Assignment ON Time)
- [Problem Statement](#problem-statement)
- [Results & Analysis](#results-analysis)
  * [Augumentaion Strategy Used](#augumentaion-strategy-used)
  * [Our Learnings](#our-learnings)
- [CIFAR-10 Vizualization And Augmentation](#cifar-10-vizualization-and-augmentation)
- [Our Model](#our-model)
  * [Training Log](#training-log)
- [Model Evaluation](#model-evaluation)
  * [Learning Curve](#learning-curve)
  * [Missclassified Images](#misclassified-images)
  * [Accuracy Of Each Class](#accuracy-of-each-class)
- [Refrences](#refrences)
- [Team Members](#team-members)

# Problem Statement

- - Train for 40 Epochs
  - **20** misclassified images
  - **20** GradCam output on the **SAME misclassified images**
  - Apply these transforms while training:
      1. RandomCrop(32, padding=4)
      2. CutOut(16x16)
      3. **Rotate(±5°)**
  - **Must** use ReduceLROnPlateau
  - **Must** use LayerNormalization ONLY

# Results Analysis


#### Our Learnings

# CIFAR-10 Vizualization And Augmentation

- **DataSet:** CIFAR-10 has 10 classes of 32,32 that are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

<p float="center">
  <img src="images/visualization_2.png" alt="drawing" width="700" height="550">
</p>

- Augmentation 
Effect of applying the augmentation on some of the test images.

<p float="center">
  <img src="images/visualization_1.png" alt="drawing" width="750" height="650">
</p>


# Our Model
<p float="center">
  <img src="images/RF.png" alt="drawing">
</p>

```
Total params: 86,816
Trainable params: 86,816
Non-trainable params: 0
```

### Training Log

```


```

# Model Evaluation

### Learning Curve

<p float="center">
  <img src="images/graph.png" alt="drawing" height="350">
</p>

### Misclassified Images

| |      |      |
| :------------------------------------------------------------: | :----------------------------------------------------------: | :------------------------------------------------------------: |
|      |      |      |
|      |      |      |

### Accuracy Of Each Class

```
Accuracy of airplane : 88 %
Accuracy of automobile : 96 %
Accuracy of  bird : 83 %
Accuracy of   cat : 76 %
Accuracy of  deer : 87 %
Accuracy of   dog : 78 %
Accuracy of  frog : 86 %
Accuracy of horse : 93 %
Accuracy of  ship : 93 %
Accuracy of truck : 92 %
```

Refrences
----------------



Team Members
------------------------

Neha Mittal, Vivek Chaudhary

