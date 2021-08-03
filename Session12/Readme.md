# Submission for Week 12

- [Problem Statement](#problem-statement)
- [Results](#results)
- [Model Evaluation](#model-evaluation)
  * [Training Logs](#training-logs)
  * [Misclassified Images](#misclassified-images)
  * [STN Vizulation](#stn-vizulation)
- [Team Members](#team-members)
- [Understanding ViT (Implemented on PyTorch)](#understanding-vit)


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


# Results

- Link to [Notebook](https://github.com/vivek-a81/EVA6/blob/main/Session12/S12CIFAR10transformer.ipynb)
- Link to [Main Repo](https://github.com/MittalNeha/vision_pytorch)
- Test Accuracy : 51.23%
- Train Accuracy : 60.18%

# Model Evaluation

* Last 10 training logs
* Misclassified Images
* STN Vizulation


Training Logs
--------------------------

* Here are last 10 logs
```
EPOCH: 40 (LR: 0.01)
Batch_id=195 Loss=1.40958 Accuracy=49.62%: 100%|██████████| 196/196 [00:08<00:00, 24.37it/s]
  0%|          | 0/196 [00:00<?, ?it/s]
Test set: Average loss: 1.1630, Accuracy: 6061/10000 (60.61%)

EPOCH: 41 (LR: 0.01)
Batch_id=195 Loss=1.39998 Accuracy=50.24%: 100%|██████████| 196/196 [00:08<00:00, 24.27it/s]
  0%|          | 0/196 [00:00<?, ?it/s]
Test set: Average loss: 1.1569, Accuracy: 5998/10000 (59.98%)

EPOCH: 42 (LR: 0.01)
Batch_id=195 Loss=1.40912 Accuracy=49.65%: 100%|██████████| 196/196 [00:08<00:00, 22.54it/s]
  0%|          | 0/196 [00:00<?, ?it/s]
Test set: Average loss: 1.2076, Accuracy: 5833/10000 (58.33%)

EPOCH: 43 (LR: 0.01)
Batch_id=195 Loss=1.38702 Accuracy=50.33%: 100%|██████████| 196/196 [00:08<00:00, 22.76it/s]
  0%|          | 0/196 [00:00<?, ?it/s]
Test set: Average loss: 1.1455, Accuracy: 6032/10000 (60.32%)

EPOCH: 44 (LR: 0.01)
Batch_id=195 Loss=1.38077 Accuracy=50.87%: 100%|██████████| 196/196 [00:08<00:00, 23.77it/s]
  0%|          | 0/196 [00:00<?, ?it/s]
Test set: Average loss: 1.1558, Accuracy: 5925/10000 (59.25%)

EPOCH: 45 (LR: 0.01)
Batch_id=195 Loss=1.39234 Accuracy=50.13%: 100%|██████████| 196/196 [00:08<00:00, 22.64it/s]
  0%|          | 0/196 [00:00<?, ?it/s]
Test set: Average loss: 1.4079, Accuracy: 4982/10000 (49.82%)

EPOCH: 46 (LR: 0.01)
Batch_id=195 Loss=1.41334 Accuracy=49.60%: 100%|██████████| 196/196 [00:08<00:00, 22.61it/s]
  0%|          | 0/196 [00:00<?, ?it/s]
Test set: Average loss: 1.2168, Accuracy: 5763/10000 (57.63%)

EPOCH: 47 (LR: 0.01)
Batch_id=195 Loss=1.38242 Accuracy=50.73%: 100%|██████████| 196/196 [00:08<00:00, 23.44it/s]
  0%|          | 0/196 [00:00<?, ?it/s]
Test set: Average loss: 1.1460, Accuracy: 5992/10000 (59.92%)

EPOCH: 48 (LR: 0.01)
Batch_id=195 Loss=1.38006 Accuracy=50.90%: 100%|██████████| 196/196 [00:08<00:00, 23.94it/s]
  0%|          | 0/196 [00:00<?, ?it/s]
Test set: Average loss: 1.4068, Accuracy: 4923/10000 (49.23%)

EPOCH: 49 (LR: 0.01)
Batch_id=195 Loss=1.40337 Accuracy=50.06%: 100%|██████████| 196/196 [00:08<00:00, 24.12it/s]
  0%|          | 0/196 [00:00<?, ?it/s]
Test set: Average loss: 1.1414, Accuracy: 5983/10000 (59.83%)

EPOCH: 50 (LR: 0.01)
Batch_id=195 Loss=1.36774 Accuracy=51.23%: 100%|██████████| 196/196 [00:08<00:00, 23.78it/s]
Test set: Average loss: 1.1450, Accuracy: 6018/10000 (60.18%)
```

Misclassified Images
--------------------------

<p float="center">
  <img src="images/misclfpng.png" alt="drawing" height="550">
</p>

STN Vizulation
----------------
Click to view image
<p float="center">
  <img src="images/stn.png" alt="drawing" height="850">
</p>


# Understanding ViT

The Vision Transformer (ViT) have gained a lot of traction for image recognition application in the last few years.

<Some images of sota>

[This](https://github.com/jeonsworld/ViT-pytorch) repo compares the implementation from the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) and presents the accuracy for image recognition. To understand ViT better, we will start with looking at the model used in this repo. 

Direct [link](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py) to the model file. 

As an example, we will consider the ViT-B/16 algorithm and CIFAR10 dataset. The configuration for ViT-B/16 is given as:

```
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
```

If we look at this animation: 
<p float="left">
 <img src="images/image1.gif" alt="drawing" width="650">
</p>

The main classes in this file are:

- <u>Embeddings</u>: This class is used to take the image input and give out the 9+1 embedding outputs as per the gif shown above. The output form the operations in this class is an embedding that comprises of two chief elements: patch_embeddings and position_embeddings. 
  - *patch_embeddings* are obtained by passing the input image through a Conv2d layer.
    Conv2d(in_channels=in_channels,
                                         out_channels=config.hidden_size,
                                         kernel_size=patch_size,
                                         stride=patch_size)
    The patch_size for ViT is 16x16 and for CIFAR10, the input image is 32x32x3. Hence the parameters for our example will look like:
    Conv2d(3, 768, (16,16), (16,16)). Another variable, n_patches=2x2 = 4
    <diagram to show the input and output of the convolution>
  - *position_embeddings* is a Tensor of size (1, 4+1, 768). Here the +1 is for the cls_token (classification token) which is also initialized with zeros and is then learnt along with the network.

- <u>MLP</u>: This is the class for the only layer that introduces non-linearity.
- <u>Attention:</u> In this class, there are three Linear layers, one each for query, key and value. The output from query layer is multiplied by the transpose of the output from key layer. This normalized output passes through softmax and is then multiplied by the output of value_layer.

<p float="left">
 <img src="images/attention.png" alt="drawing" width="500">
</p>

- <u>Block:</u> This class has two residual blocks.

         1. A Layer Normalization layer followed by the output of the Attention class output. This output added with the input as a residual connection.
         2. A Layer Normalization layer followed by the output of the MLP class output. 

<p float="left">
 <img src="images/block.png" alt="drawing" width="500">
</p>

- <u>Encoder:</u> Encoder is the 'num_layers' of Block appended one after the other, to create the Encoder layer.




Team Members
------------------------

Neha Mittal, Vivek Chaudhary
