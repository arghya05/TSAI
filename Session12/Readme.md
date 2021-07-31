# Submission for Week 12

- [Problem Statement](#problem-statement)
- [Results & Analysis](#results-analysis)
- [Model Evaluation](#model-evaluation)
  * [Custom-resnet Learning Curve](#Custom-resnet-learning-curve)
  * [Custom-resnet Misclassified Images](#Custom-resnet-misclassified-images)
- [Team Members](#team-members)


# Problem Statement


- Increase the number of images annotated for this week from 50 to 100. 
- Implement that [Spatial Transformer Code](https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html) for CIFAR10 and submit. Must have proper readme and must be trained for 50 Epochs. Proceed to Assignment QnA and submit
- a link to README on what Spatial Transfomer does. The readme must also have the link to your Google Colab file (public). 
- link to your notebook on GitHub. 
- link to another readme where you'll describe using text and your drawn images, the classes in this [file](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py):
  - Block
  - Embeddings
  - MLP
  - Attention
  - Encoder


# Results Analysis
Link to [Notebook](https://github.com/vivek-a81/EVA6/blob/main/Session9/Session9.ipynb)

Link to [Main Repo](https://github.com/MittalNeha/vision_pytorch)
- Test Accuracy : 88.66%
- Train Accuracy : 94.34%
- LR finder was used to find the best accuracy and used as the lr_max in OneCycleLR
- Adding L2 Regularisation boosted the performance of the model  
LR Finder plot
<p float="center">
  <img src="images/lr-finder.png" alt="drawing" width="450" height="350">
</p>
Augmentation Strategy Used
```
     A.Sequential([
                   A.CropAndPad(px=4, keep_size=False), #padding of 2, keep_size=True by defaulf
                   A.RandomCrop(32,32)
                   ]),
     A.HorizontalFlip(),
     A.CoarseDropout(1, 8, 8, 1, 8, 8,fill_value=0.473363, mask_fill_value=None),
     A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
```

<p float="center">
  <img src="images/lr-model-1.png" alt="drawing" width="350" height="250">
</p>


# Model Evaluation

We have plotted
* Custom resnet Learning Curve
* Custom resnet Misclassified Images


Custom resnet Learning Curve
--------------------------

<p float="center">
  <img src="images/ler_cur.png" alt="drawing" width="750">
</p>


Custom resnet Misclassified Images
--------------------------

<p float="center">
  <img src="images/mis_clf.png" alt="drawing" height="550">
</p>


References
------------------------

* https://github.com/davidtvs/pytorch-lr-finder
* https://stackoverflow.com/questions/54553388/have-i-implemented-implemenation-of-learning-rate-finder-correctly
* https://discuss.pytorch.org/t/get-the-best-learning-rate-automatically/58269


Team Members
------------------------

Neha Mittal, Vivek Chaudhary
