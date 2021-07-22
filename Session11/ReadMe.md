# Submission for Week 11

- [Problem Statement](#problem-statement)
- [Assignment A](#assignment-a)
- [Results & Analysis](#results-analysis)
- [Data Vizulation](#tiny-image-net)
- [Tiny Image Net 200 Augmentation Vizulation](#tiny-imagenet-200-augmentation-vizulation)
- [Model Evaluation](#model-evaluation)
  * [Custom-resnet Learning Curve](#resnet18-learning-curve)
  * [Custom-resnet Misclassified Images](#resnet18-misclassified-images)
  * [Training Logs Last 10 epcoh](#training-logs)
- [Assignment B](#assignment-b)
- [Problem Statement of Assignment B](#problem-statement-of-assignment-b)
- [COCO DATASET](#coco-dataset)
- [Refrences](#references)
- [Team Members](#team-members)


# Problem Statement
1. Assignment A:
   1. Download this [TINY IMAGENET ](http://cs231n.stanford.edu/tiny-imagenet-200.zip)dataset. 
   2. Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. 
   3. Submit Results. Of course, you are using your own package for everything. You can look at [this ](https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/Train_ResNet_On_Tiny_ImageNet.ipynb) for reference. 
2. Assignment B:
   1.  Learn how COCO object detection dataset's schema is. This file has the same schema. You'll need to discover what those number are. 
   2. Identify these things for this dataset:
      1. readme data for class distribution (along with the class names) along with a graph 
      2. Calculate the Anchor Boxes for k = 3, 4, 5, 6 and draw them.


# Assignment A

- Link to [Notebook](https://github.com/vivek-a81/EVA6/blob/main/Session10/Assignment%20A/tiny_imagenet.ipynb)
- Link to [Main Repo](https://github.com/MittalNeha/vision_pytorch)

## Results Analysis

```
Augmentation Used:
  A.PadIfNeeded(min_height=76, min_width=76, always_apply=True),
  A.RandomCrop(64,64),
  A.Rotate(limit=15),
  A.CoarseDropout(1,24, 24, 1, 8, 8,fill_value=mean*255., mask_fill_value=None),
  A.VerticalFlip(),
  A.HorizontalFlip(),
  A.Normalize(mean, std),
  ToTensorV2()
```
- Dataset Used: tiny ImageNet
- Test Accuracy : 50.13% in 50 epochs reached at 44 epoch
- Train Accuracy : 95.26%


One Cycle Policy
```
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=1.0,
                                                steps_per_epoch=len(train_loader), 
                                                epochs=50,
                                                pct_start=0.2,
                                                div_factor=10,
                                                three_phase=True, 
                                                )
```                                            


## Tiny Image Net

- **DataSet:** tiny Imagenet has **200 classes** of **64x64** images. Each class has 500 images in the training dataset

<p float="center">
  <img src="images/data.png" alt="drawing" width="650">
</p>

## Tiny ImageNet 200 Augmentation Vizulation

<p float="center">
  <img src="images/aug.png" alt="drawing" width="650">
</p>


# Model Evaluation

We have plotted
* ResNet18 Learning Curve
* ResNet18 Misclassified Images


ResNet18 Learning Curve
--------------------------

<p float="center">
  <img src="images/lr.png" alt="drawing" width="650">
</p>


ResNet18 Misclassified Images
--------------------------

<p float="center">
  <img src="images/misclf.png" alt="drawing" height="600">
</p>

Training Logs
----------

```
EPOCH: 41 (LR: 0.024985442450470904)
Batch_id=273 Loss=0.26415 Accuracy=92.52%: 100%|██████████| 274/274 [04:46<00:00,  1.05s/it]
  0%|          | 0/274 [00:00<?, ?it/s]
Test set: Average loss: 3.1290, Accuracy: 14962/30000 (49.87%)

EPOCH: 42 (LR: 0.02059858656808367)
Batch_id=273 Loss=0.23973 Accuracy=93.07%: 100%|██████████| 274/274 [04:46<00:00,  1.05s/it]
  0%|          | 0/274 [00:00<?, ?it/s]
Test set: Average loss: 3.1563, Accuracy: 14925/30000 (49.75%)

EPOCH: 43 (LR: 0.01653383520088566)
Batch_id=273 Loss=0.22410 Accuracy=93.72%: 100%|██████████| 274/274 [04:46<00:00,  1.05s/it]
  0%|          | 0/274 [00:00<?, ?it/s]
Test set: Average loss: 3.1600, Accuracy: 14990/30000 (49.97%)

EPOCH: 44 (LR: 0.012835711791748522)
Batch_id=273 Loss=0.21369 Accuracy=94.08%: 100%|██████████| 274/274 [04:46<00:00,  1.05s/it]
  0%|          | 0/274 [00:00<?, ?it/s]
Test set: Average loss: 3.1778, Accuracy: 15018/30000 (50.06%)

EPOCH: 45 (LR: 0.009544723907178754)
Batch_id=273 Loss=0.20005 Accuracy=94.58%: 100%|██████████| 274/274 [04:46<00:00,  1.05s/it]
  0%|          | 0/274 [00:00<?, ?it/s]
Test set: Average loss: 3.1830, Accuracy: 15046/30000 (50.15%)

EPOCH: 46 (LR: 0.006696919535806679)
Batch_id=273 Loss=0.19192 Accuracy=94.86%: 100%|██████████| 274/274 [04:46<00:00,  1.05s/it]
  0%|          | 0/274 [00:00<?, ?it/s]
Test set: Average loss: 3.2010, Accuracy: 15041/30000 (50.14%)

EPOCH: 47 (LR: 0.004323492235068956)
Batch_id=273 Loss=0.18853 Accuracy=94.83%: 100%|██████████| 274/274 [04:47<00:00,  1.05s/it]
  0%|          | 0/274 [00:00<?, ?it/s]
Test set: Average loss: 3.1962, Accuracy: 15045/30000 (50.15%)

EPOCH: 48 (LR: 0.002450439451128521)
Batch_id=273 Loss=0.18120 Accuracy=95.06%: 100%|██████████| 274/274 [04:46<00:00,  1.05s/it]
  0%|          | 0/274 [00:00<?, ?it/s]
Test set: Average loss: 3.1868, Accuracy: 15040/30000 (50.13%)

EPOCH: 49 (LR: 0.0010982777546406704)
Batch_id=273 Loss=0.18038 Accuracy=95.19%: 100%|██████████| 274/274 [04:46<00:00,  1.05s/it]
  0%|          | 0/274 [00:00<?, ?it/s]
Test set: Average loss: 3.1999, Accuracy: 15038/30000 (50.13%)

EPOCH: 50 (LR: 0.000281818111543607)
Batch_id=273 Loss=0.17845 Accuracy=95.26%: 100%|██████████| 274/274 [04:46<00:00,  1.05s/it]
Test set: Average loss: 3.2014, Accuracy: 15038/30000 (50.13%)
```

# Assignment B

- To [NoteBook](https://github.com/vivek-a81/EVA6/blob/main/Session10/Assignment%20B/Assignment_10_B.ipynb)
- To [Excel Sheet](https://github.com/vivek-a81/EVA6/blob/main/Session10/Assignment%20B/COCODataset.xlsx)

# Problem Statement of Assignment B

- Learn how COCO object detection dataset's schema is. This file has the same schema. You'll need to discover what those number are.
- Identify these things for this dataset:
     1. readme data for class distribution (along with the class names) along with a graph
     2. Calculate the Anchor Boxes **using K Means** for k = 3, 4, 5, 6 and draw them.

# COCO DATASET
The sample_coco.txt given for this assignment looks like this: id: 0, height: 330, width: 1093, bbox:[69, 464, 312, 175], id: 1, height: 782, width: 439, bbox:[359, 292, 83, 199], id: 3, height: 645, width: 831, bbox:[297, 312, 267, 167], id: 34, height: 943, width: 608, bbox:[275, 112, 319, 290], id: 20, height: 593, width: 857, bbox:[71, 368, 146, 147], id: 61, height: 587, width: 745, bbox:[177, 463, 68, 302],

Here is is the class id, followed by the height and width of the image. the bounding box (bbox). The COCO dataset defines the bounding box as x,y, width, height, where x and y is the vertex closer to origin.

<p float="center">
  <img src="Assignment B/images/anchorboxes.png" alt="drawing" width="650">
</p>


<p float="center">
  <img src="Assignment B/images/excel_ss.png" alt="drawing" width="650">
</p>

References
------------------------

* https://github.com/sonugiri1043/Train_ResNet_On_Tiny_ImageNet/blob/master/Train_ResNet_On_Tiny_ImageNet.ipynb
* https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch


Team Members
------------------------

Neha Mittal, Vivek Chaudhary
