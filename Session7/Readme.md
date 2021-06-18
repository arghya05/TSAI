# Submission for Week 7 Late Assignment ON Time


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
EPOCH: 90

Batch_id=781 Loss=0.57727 Accuracy=79.78: 100%|██████████| 782/782 [00:05<00:00, 140.26it/s]
  0%|          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.3919, Accuracy: 8689/10000 (86.89%)

EPOCH: 91

Batch_id=781 Loss=0.57028 Accuracy=79.96: 100%|██████████| 782/782 [00:05<00:00, 141.06it/s]
  0%|          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.3909, Accuracy: 8701/10000 (87.01%)

EPOCH: 92

Batch_id=781 Loss=0.56919 Accuracy=80.24: 100%|██████████| 782/782 [00:05<00:00, 139.02it/s]
  0%|          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.3909, Accuracy: 8726/10000 (87.26%)

EPOCH: 93

Batch_id=781 Loss=0.56956 Accuracy=80.03: 100%|██████████| 782/782 [00:05<00:00, 139.88it/s]
  0%|          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.3875, Accuracy: 8714/10000 (87.14%)

EPOCH: 94

Batch_id=781 Loss=0.56150 Accuracy=80.12: 100%|██████████| 782/782 [00:05<00:00, 140.19it/s]
  0%|          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.3886, Accuracy: 8706/10000 (87.06%)

EPOCH: 95

Batch_id=781 Loss=0.55917 Accuracy=80.49: 100%|██████████| 782/782 [00:05<00:00, 140.80it/s]
  0%|          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.3875, Accuracy: 8695/10000 (86.95%)

EPOCH: 96

Batch_id=781 Loss=0.55844 Accuracy=80.41: 100%|██████████| 782/782 [00:05<00:00, 140.66it/s]
  0%|          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.3883, Accuracy: 8712/10000 (87.12%)

EPOCH: 97

Batch_id=781 Loss=0.56086 Accuracy=80.50: 100%|██████████| 782/782 [00:05<00:00, 140.29it/s]
  0%|          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.3884, Accuracy: 8712/10000 (87.12%)

EPOCH: 98

Batch_id=781 Loss=0.55907 Accuracy=80.67: 100%|██████████| 782/782 [00:05<00:00, 140.17it/s]
  0%|          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.3868, Accuracy: 8710/10000 (87.10%)

EPOCH: 99

Batch_id=781 Loss=0.55661 Accuracy=80.51: 100%|██████████| 782/782 [00:05<00:00, 141.24it/s]
  0%|          | 0/782 [00:00<?, ?it/s]

Test set: Average loss: 0.3873, Accuracy: 8725/10000 (87.25%)

EPOCH: 100

Batch_id=781 Loss=0.55942 Accuracy=80.51: 100%|██████████| 782/782 [00:05<00:00, 140.16it/s]

Test set: Average loss: 0.3876, Accuracy: 8723/10000 (87.23%)

```

# Model Evaluation

### Learning Curve

<p float="center">
  <img src="images/graph.png" alt="drawing" height="350">
</p>

### Missclassified Images
<p float="center">
  <img src="images/misclassify1.png" alt="drawing" height="1050">
</p>


Team Members
------------------------

Neha Mittal, Vivek Chaudhary

