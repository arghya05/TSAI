# Problem Statement

Write a network architecture in such a way:
- 99.4% validation accuracy
- Less than 20k Parameters
- You can use anything from above you want. 
- Less than 20 Epochs
- Have used BN, Dropout, a Fully connected layer, have used GAP. 


## Network Architecture:

![s4arcg](https://user-images.githubusercontent.com/84603388/120033045-b1d1d900-c018-11eb-821c-ef2a2a6416a4.png)

[Click to notebook](https://github.com/vivek-a81/EVA6/blob/main/Session%204/Part-2/Session_4_Assignment.ipynb)

- There are total 17266 parameters in networks.
- Dropout of 0.1 is used every layer (except after fully connected layer).
- Achieved validation accuracy of 99.5% and 99.40% at epoch 12.
- Model was trained with SGD with learning rate of 0.05 and momentum 0.9.

## Model Summary:

          Layer (type)               Output Shape         Param #
              Conv2d-1            [-1, 8, 28, 28]              72
                ReLU-2            [-1, 8, 28, 28]               0
         BatchNorm2d-3            [-1, 8, 28, 28]              16
              Conv2d-4           [-1, 16, 28, 28]           1,152
                ReLU-5           [-1, 16, 28, 28]               0
         BatchNorm2d-6           [-1, 16, 28, 28]              32
              Conv2d-7           [-1, 24, 28, 28]           3,456
                ReLU-8           [-1, 24, 28, 28]               0
         BatchNorm2d-9           [-1, 24, 28, 28]              48
          MaxPool2d-10           [-1, 24, 14, 14]               0
          Dropout2d-11           [-1, 24, 14, 14]               0
             Conv2d-12            [-1, 8, 14, 14]             192
               ReLU-13            [-1, 8, 14, 14]               0
             Conv2d-14           [-1, 16, 12, 12]           1,152
               ReLU-15           [-1, 16, 12, 12]               0
        BatchNorm2d-16           [-1, 16, 12, 12]              32
             Conv2d-17           [-1, 32, 10, 10]           4,608
               ReLU-18           [-1, 32, 10, 10]               0
        BatchNorm2d-19           [-1, 32, 10, 10]              64
          Dropout2d-20           [-1, 32, 10, 10]               0
             Conv2d-21            [-1, 8, 10, 10]             256
               ReLU-22            [-1, 8, 10, 10]               0
             Conv2d-23             [-1, 16, 8, 8]           1,152
               ReLU-24             [-1, 16, 8, 8]               0
        BatchNorm2d-25             [-1, 16, 8, 8]              32
             Conv2d-26             [-1, 32, 6, 6]           4,608
               ReLU-27             [-1, 32, 6, 6]               0
        BatchNorm2d-28             [-1, 32, 6, 6]              64
          Dropout2d-29             [-1, 32, 6, 6]               0
          AvgPool2d-30             [-1, 32, 1, 1]               0
             Linear-31                   [-1, 10]             330
  Total params: 17,266
  Trainable params: 17,266
  Non-trainable params: 0


## Learning Curve:

![Screenshot (272)](https://user-images.githubusercontent.com/84603388/120034109-2bb69200-c01a-11eb-8502-683c1b823927.png)

## Model Evaluation of test set:

![output11](https://user-images.githubusercontent.com/84603388/120034196-5274c880-c01a-11eb-99e6-a3ee5ea2e62d.png)

## Training logs:

  Epoch:1
  loss=0.0445 batch_id=468: 100%
  469/469 [04:49<00:00, 1.62it/s]

  Test set: Average loss: 0.0546, Accuracy: 9833/10000 (98.33%)

  Epoch:2
  loss=0.1304 batch_id=468: 100%
  469/469 [00:14<00:00, 32.33it/s]

  Test set: Average loss: 0.0544, Accuracy: 9833/10000 (98.33%)

  Epoch:3
  loss=0.0462 batch_id=468: 100%
  469/469 [04:20<00:00, 1.80it/s]

  Test set: Average loss: 0.0334, Accuracy: 9890/10000 (98.90%)

  Epoch:4
  loss=0.1865 batch_id=468: 100%
  469/469 [00:14<00:00, 32.54it/s]

  Test set: Average loss: 0.0322, Accuracy: 9892/10000 (98.92%)

  Epoch:5
  loss=0.0306 batch_id=468: 100%
  469/469 [03:51<00:00, 2.03it/s]

  Test set: Average loss: 0.0348, Accuracy: 9887/10000 (98.87%)

  Epoch:6
  loss=0.0139 batch_id=468: 100%
  469/469 [00:15<00:00, 30.85it/s]

  Test set: Average loss: 0.0258, Accuracy: 9924/10000 (99.24%)

  Epoch:7
  loss=0.0373 batch_id=468: 100%
  469/469 [03:22<00:00, 2.32it/s]

  Test set: Average loss: 0.0252, Accuracy: 9916/10000 (99.16%)

  Epoch:8
  loss=0.0310 batch_id=468: 100%
  469/469 [00:15<00:00, 30.64it/s]

  Test set: Average loss: 0.0255, Accuracy: 9916/10000 (99.16%)

  Epoch:9
  loss=0.0013 batch_id=468: 100%
  469/469 [02:53<00:00, 2.70it/s]

  Test set: Average loss: 0.0272, Accuracy: 9913/10000 (99.13%)

  Epoch:10
  loss=0.0980 batch_id=459: 98%
  469/469 [00:15<00:00, 30.97it/s]
  Test set: Average loss: 0.0261, Accuracy: 9924/10000 (99.24%)

  Epoch:11
  loss=0.0124 batch_id=468: 100%
  469/469 [02:24<00:00, 3.24it/s]

  Test set: Average loss: 0.0208, Accuracy: 9935/10000 (99.35%)

  Epoch:12
  loss=0.0149 batch_id=468: 100%
  469/469 [00:14<00:00, 32.35it/s]

  Test set: Average loss: 0.0198, Accuracy: 9940/10000 (99.40%)


 
