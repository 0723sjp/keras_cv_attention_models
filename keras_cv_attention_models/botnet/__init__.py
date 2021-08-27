from keras_cv_attention_models.botnet.botnet import BotNet, BotNet50, BotNet101, BotNet152, MHSAWithPositionEmbedding


__head_doc__ = """
Keras implementation of [botnet](https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2).
Paper [PDF 2101.11605 Bottleneck Transformers for Visual Recognition](https://arxiv.org/pdf/2101.11605.pdf).
"""

__tail_doc__ = """  strides: strides used in the last stack. It's reported `1` works better than `2`, but slower.
  preact: whether to use pre-activation or not.
  use_bias: whether to use biases for convolutional layers or not.
  input_shape: it should have exactly 3 inputs channels, default `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

BotNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model        | Params |
  | ------------ | ------ |
  | botnet50     | 21M    |
  | botnet101    | 41M    |
  | botnet152    | 56M    |
"""

BotNet50.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

BotNet101.__doc__ = BotNet50.__doc__
BotNet152.__doc__ = BotNet50.__doc__

MHSAWithPositionEmbedding.__doc__ = __head_doc__ + """
Multi head self attention with positional embedding layer.

Args:
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key.
  relative: Boolean, whether using relative or absolute positional embedding.
  out_weight: Boolean, whether use an ouput dense.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  out_shape: The expected shape of an output tensor. If not specified, projects back to the key feature dim.
  attn_dropout: Dropout probability for attention.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.MHSAWithPositionEmbedding(num_heads=4, key_dim=128)
>>> print(f"{aa(tf.ones([1, 14, 16, 256])).shape = }")
aa(tf.ones([1, 14, 16, 256])).shape = TensorShape([1, 14, 16, 512])

>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
{'mhsa_with_position_embedding_2/query:0': (256, 512),
 'mhsa_with_position_embedding_2/key:0': (256, 512),
 'mhsa_with_position_embedding_2/value:0': (256, 512),
 'mhsa_with_position_embedding_2/output_weight:0': (512, 512),
 'mhsa_with_position_embedding_2/r_width:0': (128, 31),
 'mhsa_with_position_embedding_2/r_height:0': (128, 27)}
"""
