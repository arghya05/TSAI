# Understanding ViT (Implemented on PyTorch)

The Vision Transformer (ViT) have gained a lot of traction for image recognition application in the last few years.

<Some images of sota>

[This](https://github.com/jeonsworld/ViT-pytorch) repo compares the implementation from the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) and presents the accuracy for image recognition. To understand ViT better, we will start with looking at the model used in this repo. 

Direct [link](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py) to the model file. 

The main classes in this file are:

- Block
- Embeddings
- MLP
- Attention
- Encoder

