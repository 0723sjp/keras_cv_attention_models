from keras_cv_attention_models.moganet.moganet import MogaNet, MogaNetXtiny, MogaNetTiny, MogaNetSmall, MogaNetBase, MogaNetLarge

__head_doc__ = """
Keras implementation of [Github Westlake-AI/MogaNet](https://github.com/Westlake-AI/MogaNet).
Paper [PDF 2211.03295 Efficient Multi-order Gated Aggregation Network](https://arxiv.org/pdf/2211.03295.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation for non-attention blocks, default `gelu`.
  attn_activation: activation for attention blocks, default `swish`. `None` for same with `activation`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  layer_scale: int value indicates layer scale init value for each stack. Default `1e-5`, 0 for not using.
      [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

MogaNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  mlp_ratios: int or list value indicates expand ratio for mlp blocks hidden channel in each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model        | Params | FLOPs  | Input | Top1 Acc |
  | ------------ | ------ | ------ | ----- | -------- |
  | MogaNetXtiny | 2.96M  | 806M   | 224   | 76.5     |
  | MogaNetTiny  | 5.20M  | 1.11G  | 224   | 79.0     |
  |              | 5.20M  | 1.45G  | 256   | 79.6     |
  | MogaNetSmall | 25.3M  | 4.98G  | 224   | 83.4     |
  | MogaNetBase  | 43.7M  | 9.96G  | 224   | 84.2     |
  | MogaNetLarge | 82.5M  | 15.96G | 224   | 84.6     |
"""

MogaNetXtiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

MogaNetTiny.__doc__ = MogaNetXtiny.__doc__
MogaNetSmall.__doc__ = MogaNetXtiny.__doc__
MogaNetBase.__doc__ = MogaNetXtiny.__doc__
MogaNetLarge.__doc__ = MogaNetXtiny.__doc__
