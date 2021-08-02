# Understanding ViT (Implemented on PyTorch)

The Vision Transformer (ViT) have gained a lot of traction for image recognition application in the last few years.

<Some images of sota>

[This](https://github.com/jeonsworld/ViT-pytorch) repo compares the implementation from the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) and presents the accuracy for image recognition. To understand ViT better, we will start with looking at the model used in this repo. 

Direct [link](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py) to the model file. 

As an example, we will consider the ViT-B/16 algorithm and CIFAR10 dataset. The configuratiion for ViT-B/16 is given as:
    ```config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None```
  
If we look at this annimation: 
  .gif

The main classes in this file are:

- Block
- Embeddings: This class is used to take the image input and give out the 9+1 enbedding outputs as per the gif shown above. The output form the operations in this class is an embedding that comprices of two chief elements: patch_embeddings and position_embeddings. 
patch_embeddings are obtained by passing the input image through a Conv2d layer.
  Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
The patch_size for ViT is 16x16 and for CIFAR10, the input image is 32x32x3. Hence the parameters for our example will look like:
Conv2d(3, 768, (16,16), (16,16)). Another variable, n_patches=2x2 = 4
  <diagram to show the input and output of the convolution>
    
position_embeddings is a Tensor of size (1,4+1,768). Here the +1 is for the cls_token(classification token) which is also initialized with zeros and isthen learnt along with the network.
    
- MLP
- Attention
- Encoder

