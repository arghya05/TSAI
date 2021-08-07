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
 <img src="https://github.com/vivek-a81/EVA6/blob/main/Session13/images/vit-01.png" alt="drawing">
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
 <img src="https://github.com/vivek-a81/EVA6/blob/main/Session13/images/visualizing-positional-encodings-vit.png" alt="drawing">
</p>

The configuration values are also inputed to class which contains all the information regrading input size, dropout value, hidden state, embeddings etc.
The input image is converted to patch embeddings then class embeddings are concatenated to each patch in the image of a particular batch size after that 
the positional embeddings are added to each and every patch in the patch.

<p float="center">
 <img src="https://github.com/vivek-a81/EVA6/blob/main/Session13/images/vit-03.png" alt="drawing">
</p>

### Transformer Encoder

<p float="center">
 <img src="https://github.com/vivek-a81/EVA6/blob/main/Session13/images/vit-07.png" alt="drawing">
</p>

Transformer encoder works in the following steps:
* Send the embeddings to the layer normalization
* Generate key, value & quaries from the embeddings
* To get the attention scores multiply quary with tanspose of key
* Divide the attention scores with square root of attention head size and pass it through softmax layer to get the attention probabs
* Multiply the attention probablities with the values to get the context layer

```
class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
```

ViTSelfOutput is a class for adding a linear layer to the output of context layer, we add linear to transforn the context layer so the in_features and out_features are same

```
class ViTSelfOutput(nn.Module):
  """
  This is just a Linear Layer Block
  """
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)

    return hidden_states
```

ViTAttention combined two calsses ViTSelfAttention & ViTSelfOutput the input states of embeddings are first send to the ViTSelfAttention to get the context layer and then to 
ViTSelfOutput, later the input is added to the output of ViTSelfOutput making a skip connnection 

```
class ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
```
A complete Transformer Self Attention Encoder layer is created by combining all the classes above followed by layer normalization then a MLP head

```
class ViTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)

        return hidden_states
class ViTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states
```
Above two classes form a MLP head
```
class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        return layer_output
  ```
  
### VIT Model
  
There are 12 such layer in a vit model 

```
class ViTModel():
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings


    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import ViTFeatureExtractor, ViTModel
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
            >>> model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return sequence_output,pooled_output,encoder_outputs.hidden_states,encoder_outputs.attentions
```
The 0-th index output i.e. [CLS] token are only taken for further processing after the 12 encoder blocks as all information is pushed towards this. A linear layer is added
to map the output to the number of classes for classification.

```
sequence_output = encoder_output[0]
layernorm = nn.LayerNorm(config.hidden_size, eps=0.00001)
sequence_output = layernorm(sequence_output)
# VitPooler
dense = nn.Linear(config.hidden_size, config.hidden_size)
activation = nn.Tanh()
first_token_tensor = sequence_output[:, 0]
pooled_output = dense(first_token_tensor)
pooled_output = activation(pooled_output)

classifier = nn.Linear(config.hidden_size, 100)
logits = classifier(pooled_output)
```


# Refrences

- https://jacobgil.github.io/deeplearning/vision-transformer-explainability
- https://arxiv.org/abs/2010.11929
- https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html
- https://youtu.be/TQQlZhbC5ps
- https://www.youtube.com/watch?v=HZ4j_U3FC94
- https://www.youtube.com/watch?v=dichIcUZfOw
- https://arxiv.org/pdf/2104.11178v1.pdf
- https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
