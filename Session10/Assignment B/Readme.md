# Week 10 - Assignment B

- [Problem Statement](#problem-statement)
- [Results & Analysis](#results-analysis)
- [CIFAR-10 Augmentation Vizulation](#cifar-10-augmentation-vizualization)
- [Model Evaluation](#model-evaluation)
  * [Custom-resnet Learning Curve](#Custom-resnet-learning-curve)
  * [Custom-resnet Misclassified Images](#Custom-resnet-misclassified-images)
- [Team Members](#team-members)


# Problem Statement
1.  Learn how COCO object detection dataset's schema is. This file has the same schema. You'll need to discover what those number are. 
2. Identify these things for this dataset:
   1. readme data for class distribution (along with the class names) along with a graph 
   2. Calculate the Anchor Boxes for k = 3, 4, 5, 6 and draw them.


# Dataset - COCO
The sample_coco.txt given for this assignment looks like this:
id: 0, height: 330, width: 1093, bbox:[69, 464, 312, 175],

id: 1, height: 782, width: 439, bbox:[359, 292, 83, 199],

id: 3, height: 645, width: 831, bbox:[297, 312, 267, 167],

id: 34, height: 943, width: 608, bbox:[275, 112, 319, 290],

id: 20, height: 593, width: 857, bbox:[71, 368, 146, 147],

id: 61, height: 587, width: 745, bbox:[177, 463, 68, 302],




Here id is the class id, followed by the height and width of the image. the bounding box (bbox). The COCO dataset defines the bounding box as x,y, width, height, where x and y is the vertex closer to origin.

This text file was imported in an excel and to calculate the normalized values for the bounding box. Link to the excel file is : []

<screenshot of the file>

# Finding the Anchor Boxes

K means algorithm is used to find clusters in order to define Anchor boxes to be used for classification. 

Link to [Notebook](https://github.com/vivek-a81/EVA6/blob/main/Session10/Assignment%20B/Assignment_10_B.ipynb)



References
------------------------

* https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch


Team Members
------------------------

Neha Mittal, Vivek Chaudhary

