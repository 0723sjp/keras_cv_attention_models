# ___Keras LeViT___
***

## Summary
  - [Github facebookresearch/LeViT](https://github.com/facebookresearch/LeViT).
  - LeViT article: [PDF 2104.01136 LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/pdf/2104.01136.pdf).
## Models
  | Model     | Params | Image resolution | Top1 Acc | ImageNet |
  | --------- | ------ | ---------------- | -------- | -------- |
  | LeViT128S | 7.8M   | 224              | 76.6     | [levit128s_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit128s_imagenet.h5) |
  | LeViT128  | 9.2M   | 224              | 78.6     | [levit128_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit128_imagenet.h5) |
  | LeViT192  | 11M    | 224              | 80.0     | [levit192_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit192_imagenet.h5) |
  | LeViT256  | 19M    | 224              | 81.6     | [levit256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit256_imagenet.h5) |
  | LeViT384  | 39M    | 224              | 82.6     | [levit384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit384_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import levit

  # Will download and load pretrained imagenet weights.
  mm = levit.LeViT128(pretrained="imagenet", use_distillation=True, classifier_activation=None)
  print(mm.output_names, mm.output_shape)
  # ['head', 'distill_head'] [(None, 1000), (None, 1000)]

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0))
  pred = tf.nn.softmax((pred[0] + pred[1]) / 2).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.962495), ('n02123159', 'tiger_cat', 0.008833298), ...]
  ```
  set `use_distillation=False` for output only one head.
  ```py
  mm = levit.LeViT192(use_distillation=False, classifier_activation="softmax")
  print(mm.output_names, mm.output_shape)
  # ['head'] (None, 1000)
  ```
  **Change input resolution**
  ```py
  # Will download and load pretrained imagenet weights.
  mm = levit.LeViT256(input_shape=(320, 320, 3), pretrained="imagenet", use_distillation=True, classifier_activation=None)
  # >>>> Load pretraind from: /home/leondgarse/.keras/models/levit256_imagenet.h5
  # WARNING:tensorflow:Skipping loading of weights for layer stack1_block1_attn_pos due to mismatch in shape ((400, 4) vs (196, 4)).
  # ...
  # >>>> Reload mismatched PositionalEmbedding weights: 224 -> 320
  # >>>> Reload layer: stack1_block1_attn_pos
  # ...

  # Run predict again using (320, 320)
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0))
  pred = tf.nn.softmax((pred[0] + pred[1]) / 2).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.81994164), ('n02123159', 'tiger_cat', 0.1425549), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch LeViT_128 """
  import torch
  sys.path.append('../LeViT')
  import levit as torch_levit
  torch_model = torch_levit.LeViT_128(pretrained=True)
  torch_model.eval()

  input_shape = 224
  inputs = np.random.uniform(size=(1, input_shape, input_shape, 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras LeViT_128 """
  from keras_cv_attention_models import levit
  mm = levit.LeViT128(pretrained="imagenet", use_distillation=True, classifier_activation=None)
  pred = mm(inputs)
  keras_out = ((pred[0] + pred[1]) / 2).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
***
