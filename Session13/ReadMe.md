# Vision Transformers (ViT)

Self-attention-based architectures, in particular Transformers, have become the model of choice in natural language processing (NLP). The dominant approach is to pre-train on
a large text corpus and then fine-tune on a smaller task-specific dataset (Devlin et al., 2019). Thanks to Transformers’ computational efficiency and scalability.
In computer vision, however, convolutional architectures remain dominant. Inspired by NLP successes, multiple works try 
combining CNN-like architectures with self-attention. **AN IMAGE IS WORTH 16X16 WORDS:
TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE**


VIT Baby Steps Working Procedure
---------------------

The overall architecture can be described easily in five simple steps:
- Split the input image to patches
- Flatten all the patches 
- Add positional embedding and [CLS] token to each Patch Embeddings
- Produce  lower-dimensional linear embeddings
- Pass it through a Transofrmer Encoder 
- Pass the representations of [CLS] tokens through an MLP Head to get final class predictions. 

<p float="center">
 <img src="images/vit-01.png" alt="drawing">
</p>



Step By Step Code Flow:
----------------------

### Patch Embeddings

```
class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        # FIXME look at relaxing size constraints
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x
```
**Working:** Transformers take a 1D sequence of image patches. This class is used to pass the image and convert it not these patches through the same linear projection layer.
Other way of of doing this might be to take the whole image and loop it to num_patches while slicing according to pixel size of each patch and flattening it. But this will be 
time consuming, instead a smarter and easier way would be to use convolution. Since it a conv layer it makes these projection learnable.

If `image_size=224, patch_size=16, num_channels=3, embed_dim=768`, here each patch would be of 16X16 that would make 
`num_patches=image_size/path_size = 224/16 = 14*14 = 196 patches`  `to_2tuple()` will check if the input is iterable or not if not it will make it a tuple and return it. 
`self.projection` will project the input image to `num_pathces*embed_dim`, `embedding dimmensions` are calculated by `patch_size*patch_size*channel=16*16*3=768`. We are taking 
strides equal to number of patches so that 16*16 kernal convolves on the patch once and will move to next patch.

### Position and CLS Embeddings

```
class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
```
**Working:** In transformers we have position embeddings and [CLS] token or embeddings which are again a learnable parameters. The [CLS] vector gets computed using 
self-attention, so it can only collect the relevant information from the rest of the hidden states. So, in some sense the [CLS] vector is also an average over token vectors. 
Later it the only vector used for all the processing . [CLS] token is a created using nn.Parameter of size batch_sizeX1X768.
**Possitional Embeddings** contains the imformation regarding the position and order of patches. As image is taken as simultaneously flows of patches 
through the Transformer’s encoder/decoder stack, The model itself doesn’t have any sense of position/order, there’s need to learn the position in order for the model to work.
After the low-dimensional linear projection, a trainable position embedding is added to the patch representations. 
It is interesting to see what these position embeddings look like after training:

<p float="center">
 <img src="images/visualizing-positional-encodings-vit.png" alt="drawing">
</p>

The configuration values are also inputed to class which contains all the information regrading input size, dropout value, hidden state, embeddings etc.
The input image is converted to patch embeddings then class embeddings are concatenated to each patch in the image of a particular batch size after that 
the positional embeddings are added to each and every patch in the patch.

<p float="center">
 <img src="images/vit-03.png" alt="drawing">
</p>

### Transformer Encoder



# Refrences

- https://jacobgil.github.io/deeplearning/vision-transformer-explainability
- https://arxiv.org/abs/2010.11929
- https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html
- https://youtu.be/TQQlZhbC5ps
- https://www.youtube.com/watch?v=j6kuz_NqkG0
- https://www.youtube.com/watch?v=HZ4j_U3FC94
- https://www.youtube.com/watch?v=dichIcUZfOw
- https://arxiv.org/pdf/2104.11178v1.pdf
- https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
