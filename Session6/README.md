# Submission for Week 6
[TOC]

# Problem Statement

You are making 3 versions of your 5th assignment's best model (or pick one from best assignments):

1. Network with Group Normalization + L1
2. Network with Layer Normalization + L2
3. Network with L1 + L2 + BN

# Our Model

Our Module class Net() takes a parameter for normalization. For example:

`model_gn = model.Net('gn').to(device)`

the values for norm can be:

'bn': Batch Normalization

'gn': Group Normalization

'ln': Layer Normalization



The norm_layer is defined using a function that decides which normalization to use

```
def normalize(x, w, h):
    if norm=='bn':
    	return nn.BatchNorm2d(x)
    elif norm=='ln':
    	return nn.LayerNorm([x,w,h])
    elif norm=='gn':
    	return nn.GroupNorm(num_groups,x)
    else:
    	return None
```

## Understanding Normalization

Batch Normalization:  

Group Normalization: 

Layer Normalization:

## Findings on Normalization

Error rate

## Sample Calculations

Below is the snapshot of the calculations performed on sample image inputs
[Link to Excel](https://github.com/vivek-a81/EVA6/blob/main/Session6/Normalization%20Calculations.xlsx)

<img src="https://github.com/vivek-a81/EVA6/blob/main/Session6/images/excel_calculations.png?raw=false" style="zoom: 60%;" />

## Train and Test Graphs

<image>



## Misclassified Images



## 



<img src="https://github.com/vivek-a81/EVA6/blob/main/Session6/images/excel_calculations.png?raw=false" style="zoom: 60%;" />

## Team Members

Neha Mittal, Vivek Chaudhary
