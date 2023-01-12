import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    BiasLayer,
    ChannelAffine,
    ClassToken,
    conv2d_no_bias,
    drop_block,
    drop_connect_rates_split,
    layer_norm,
    PositionalEmbedding,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "beit_base_patch16": {"imagenet21k-ft1k": {224: "d7102337a13a3983f3b6470de77b5d5c", 384: "76353026477c60f8fdcbcc749fea17b3"}},
    "beit_v2_base_patch16": {"imagenet21k-ft1k": {224: "d001dcb67cdda16bfdbb2873ab9b13c8"}},
    "beit_large_patch16": {
        "imagenet21k-ft1k": {224: "fce2d162e7fa4dba9a1b1fc5e1dec5ce", 384: "158934d07dd8b1e1c6b96883aa00a748", 512: "64d18088e91df243960e5830aab80a6e"}
    },
    "beit_v2_large_patch16": {"imagenet21k-ft1k": {224: "b3cee12a545bfb676f9f426ee7158d27"}},
    "eva_giant_patch14": {
        "imagenet21k-ft1k": {224: "5a475db6696d6e36ea896ec5dbd1c20d", 336: "fd8eeec10d6b6cb607ce033ea85b8e80", 560: "0ef0d2961523fb2047fbdb59cc347c17"}
    },
    "eva_large_patch14": {"imagenet21k-ft1k": {196: "bbeea886fbde4bd1c8c9876345273a99", 336: "4928faafd0177fe8f0d02dab4abc8e83"}},
    "flexivit_small": {"imagenet": {240: "efb73a97d099a491b69ebfaf8a337df8"}},
    "flexivit_base": {"imagenet": {240: "dac627debb194928db01e1b9b7a548fd"}},
    "flexivit_large": {"imagenet": {240: "6faa953227d2ef1df6758f8eb7234490"}},
}


@tf.keras.utils.register_keras_serializable(package="beit")
class MultiHeadRelativePositionalEmbedding(keras.layers.Layer):
    def __init__(self, with_cls_token=True, attn_height=-1, num_heads=-1, **kwargs):
        super(MultiHeadRelativePositionalEmbedding, self).__init__(**kwargs)
        self.with_cls_token, self.attn_height, self.num_heads = with_cls_token, attn_height, num_heads
        if with_cls_token:
            self.cls_token_len = 1
            self.cls_token_pos_len = 3
        else:
            self.cls_token_len = 0
            self.cls_token_pos_len = 0

    def build(self, attn_shape):
        # input (with_cls_token=True): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width + class_token`
        # input (with_cls_token=False): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width`
        # print(attn_shape)
        if self.attn_height == -1:
            height = width = int(tf.math.sqrt(float(attn_shape[2] - self.cls_token_len)))  # hh == ww, e.g. 14
        else:
            height = self.attn_height
            width = int(float(attn_shape[2] - self.cls_token_len) / height)
        num_heads = attn_shape[1] if self.num_heads == -1 else self.num_heads
        num_relative_distance = (2 * height - 1) * (2 * width - 1) + self.cls_token_pos_len
        # pos_shape = (num_relative_distance, num_heads)
        pos_shape = (num_heads, num_relative_distance)
        # initializer = tf.random_normal_initializer(stddev=0.02)
        self.relative_position_bias_table = self.add_weight(name="positional_embedding", shape=pos_shape, initializer="zeros", trainable=True)

        hh, ww = tf.meshgrid(range(height), range(width))  # tf.meshgrid is same with np.meshgrid 'xy' mode, while torch.meshgrid 'ij' mode
        coords = tf.stack([hh, ww], axis=-1)  # [14, 14, 2]
        coords_flatten = tf.reshape(coords, [-1, 2])  # [196, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [196, 196, 2]
        relative_coords_hh = relative_coords[:, :, 0] + height - 1
        relative_coords_ww = (relative_coords[:, :, 1] + width - 1) * (2 * height - 1)
        relative_coords = tf.stack([relative_coords_hh, relative_coords_ww], axis=-1)

        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)  # [196, 196]
        if attn_shape[3] != attn_shape[2]:
            # Choose the small values if value_block != query_block
            relative_position_index = relative_position_index[:, -(attn_shape[3] - self.cls_token_len) :]

        if self.with_cls_token:
            top = tf.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (num_relative_distance - 3)
            left = tf.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (num_relative_distance - 2)
            corner = tf.ones((1, 1), dtype=relative_position_index.dtype) * (num_relative_distance - 1)
            # print(f">>>> {top.shape = }, {left.shape = }, {corner.shape = }")
            # >>>> top.shape = TensorShape([1, 196]), left.shape = TensorShape([196, 1]), corner.shape = TensorShape([1, 1])
            left_corner = tf.concat([corner, left], axis=0)
            relative_position_index = tf.concat([top, relative_position_index], axis=0)
            relative_position_index = tf.concat([left_corner, relative_position_index], axis=1)  # [197, 197]
        self.relative_position_index = relative_position_index

    def call(self, attention_scores, **kwargs):
        pos_emb = tf.gather(self.relative_position_bias_table, self.relative_position_index, axis=1)
        # tf.print(pos_emb.shape, attention_scores.shape)
        return attention_scores + pos_emb

    def get_config(self):
        base_config = super(MultiHeadRelativePositionalEmbedding, self).get_config()
        base_config.update({"with_cls_token": self.with_cls_token, "attn_height": self.attn_height, "num_heads": self.num_heads})
        return base_config

    def load_resized_weights(self, source_layer, method="nearest"):
        if isinstance(source_layer, dict):
            source_tt = source_layer["positional_embedding:0"]  # weights
            # source_tt = source_layer["pos_emb:0"]  # weights
        else:
            source_tt = source_layer.relative_position_bias_table  # layer
        # self.relative_position_bias_table.assign(tf.transpose(source_tt))
        hh = ww = int(tf.math.sqrt(float(source_tt.shape[1] - self.cls_token_pos_len)))  # assume source weights are all square shape
        num_heads = source_tt.shape[0]
        ss = tf.reshape(source_tt[:, : hh * ww], (num_heads, hh, ww))  # [num_heads, hh, ww]
        ss = tf.transpose(ss, [1, 2, 0])  # [hh, ww, num_heads]

        if self.attn_height == -1:
            target_hh = target_ww = int(tf.math.sqrt(float(self.relative_position_bias_table.shape[1] - self.cls_token_pos_len)))
        else:
            target_hh = 2 * self.attn_height - 1
            target_ww = int(float(self.relative_position_bias_table.shape[1] - self.cls_token_pos_len) / target_hh)
        tt = tf.image.resize(ss, [target_hh, target_ww], method=method)  # [target_hh, target_ww, num_heads]
        tt = tf.reshape(tt, (tt.shape[0] * tt.shape[1], num_heads))  # [target_hh * target_ww, num_heads]
        tt = tf.transpose(tt)  # [num_heads, target_hh * target_ww]
        if self.with_cls_token:
            tt = tf.concat([tt, source_tt[:, -self.cls_token_pos_len :]], axis=1)
        self.relative_position_bias_table.assign(tt)

    def show_pos_emb(self, rows=1, base_size=2):
        import matplotlib.pyplot as plt

        num_heads = self.relative_position_bias_table.shape[0]
        # pos_emb = tf.gather(self.relative_position_bias_table, self.relative_position_index, axis=1).numpy()
        hh = ww = int(tf.math.sqrt(float(self.relative_position_bias_table.shape[1] - self.cls_token_pos_len)))
        pos_emb = tf.reshape(self.relative_position_bias_table[:, : hh * ww], (num_heads, hh, ww)).numpy()
        cols = int(tf.math.ceil(num_heads / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
        for id, ax in enumerate(axes.flatten()):
            if id >= num_heads:
                break
            ax.imshow(pos_emb[id])
            ax.set_axis_off()
        fig.tight_layout()
        return fig


def attention_block(
    inputs, num_heads=4, key_dim=0, qv_bias=True, qkv_bias=False, out_weight=True, out_bias=False, use_pos_emb=False, attn_height=-1, attn_dropout=0, name=None
):
    _, bb, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = float(1.0 / tf.math.sqrt(tf.cast(key_dim, "float32")))
    emded_dim = num_heads * key_dim

    # if qkv_bias, just use bias in qkv_dense, and set qv_bias False
    qkv_bias, qv_bias = (True, False) if qkv_bias else (False, qv_bias)

    qkv = keras.layers.Dense(emded_dim * 3, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    qkv = tf.reshape(qkv, [-1, bb, qkv.shape[-1]])
    query, key, value = tf.split(qkv, 3, axis=-1)
    # query = [batch, num_heads, cls_token + hh * ww, key_dim]
    if qv_bias:
        query = BiasLayer(name=name + "query_bias")(query)
    query = tf.reshape(query, [-1, query.shape[1], num_heads, key_dim])
    query = tf.transpose(query, [0, 2, 1, 3])
    # key = [batch, num_heads, key_dim, cls_token + hh * ww]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])
    # value = [batch, num_heads, cls_token + hh * ww, key_dim]
    if qv_bias:
        value = BiasLayer(name=name + "value_bias")(value)
    value = tf.reshape(value, [-1, value.shape[1], num_heads, key_dim])
    value = tf.transpose(value, [0, 2, 1, 3])

    query *= qk_scale
    # [batch, num_heads, cls_token + hh * ww, cls_token + hh * ww]
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key])
    if use_pos_emb:
        attention_scores = MultiHeadRelativePositionalEmbedding(attn_height=attn_height, name=name and name + "pos_emb")(attention_scores)
    # attention_scores = tf.nn.softmax(attention_scores, axis=-1, name=name and name + "_attention_scores")
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    # value = [batch, num_heads, cls_token + hh * ww, key_dim]
    # attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, cls_token + hh * ww, key_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, bb, emded_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, cls_token + hh * ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, cls_token + hh * ww, out]
        attention_output = keras.layers.Dense(emded_dim, use_bias=out_bias, name=name and name + "output")(attention_output)
    return attention_output


def attention_mlp_block(inputs, embed_dim, gamma_init_value=0.1, mlp_ratio=4, drop_rate=0, activation="gelu", attn_params={}, name=""):
    # print(f">>>> {drop_rate = }")
    nn = layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, name=name + "attn_")
    nn = attention_block(nn, **attn_params, name=name + "attn_")
    nn = ChannelAffine(use_bias=False, weight_init_value=gamma_init_value, name=name + "attn_gamma")(nn) if gamma_init_value > 0 else nn
    nn = drop_block(nn, drop_rate)
    attn_out = keras.layers.Add(name=name + "attn_out")([inputs, nn])

    """ MLP """
    nn = layer_norm(attn_out, epsilon=LAYER_NORM_EPSILON, name=name + "mlp_")
    nn = keras.layers.Dense(embed_dim * mlp_ratio, name=name + "mlp_dense_1")(nn)
    nn = activation_by_name(nn, activation, name=name + "mlp_" + activation)
    nn = keras.layers.Dense(embed_dim, name=name + "mlp_dense_2")(nn)
    nn = ChannelAffine(use_bias=False, weight_init_value=gamma_init_value, name=name + "mlp_gamma")(nn) if gamma_init_value > 0 else nn
    nn = drop_block(nn, drop_rate)
    nn = keras.layers.Add(name=name + "mlp_output")([attn_out, nn])
    return nn


@tf.keras.utils.register_keras_serializable(package="beit")
class HeadInitializer(tf.initializers.Initializer):
    def __init__(self, stddev=0.02, scale=0.001, **kwargs):
        super().__init__(**kwargs)
        self.stddev, self.scale = stddev, scale

    def __call__(self, shape, dtype="float32"):
        return tf.initializers.TruncatedNormal(stddev=self.stddev)(shape, dtype=dtype) * self.scale

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"stddev": self.stddev, "scale": self.scale})
        return base_config


@tf.keras.utils.register_keras_serializable(package="beit")
class PatchConv2DWithResampleWeights(keras.layers.Conv2D):
    def __init__(self, filters, kernel_size=1, strides=1, padding="valid", use_bias=True, groups=1, **kwargs):
        super().__init__(filters, kernel_size=kernel_size, strides=strides, padding="valid", use_bias=use_bias, groups=groups, **kwargs)
        self.padding = padding

    def build(self, input_shape):
        pad = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        if self.padding.upper() == "SAME" and max(pad) != 0:
            self.pad = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        else:
            self.pad = None
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.pad is not None:
            inputs = tf.pad(inputs, self.pad)
        return super().call(inputs)

    def load_resized_weights(self, source_layer, method="bilinear"):
        import numpy as np

        if isinstance(source_layer, dict):
            source_kernel, source_bias = source_layer["kernel:0"], source_layer["bias:0"]  # weights
        else:
            source_kernel, source_bias = source_layer.kernel, source_layer.bias  # layer

        # From FlexiViT https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py#L30
        # Paper [PDF 2212.08013 FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf)

        # assume it's channel_last format, source_kernel shape `[patch_size, patch_size, in_channel, out_channel]`
        source_kernel = np.array(source_kernel)
        source_shape, target_shape = source_kernel.shape[:2], self.kernel_size

        # get_resize_mat(old_shape, target_shape)
        # NOTE: we are using tf.image.resize here to match the resize operations in
        # the data preprocessing pipeline.
        mat = []
        for idx in range(source_shape[0] * source_shape[1]):
            basis_vec = np.zeros(source_shape)
            basis_vec[np.unravel_index(idx, source_shape)] = 1.0
            vec = tf.image.resize(np.expand_dims(basis_vec, -1), target_shape, method=method).numpy().reshape(-1)
            mat.append(vec)
        resize_mat_pinv = np.linalg.pinv(np.stack(mat))

        # v_resample_kernel = jax.vmap(jax.vmap(lambda kernel: (resize_mat_pinv @ kernel.reshape(-1)).reshape(new_hw), 2, 2), 3, 3)
        # cc = v_resample_kernel(old)
        # As it's only one weight, just using two loop here, instead of `jax.vmap`
        target_weights = np.stack([[(resize_mat_pinv @ jj.reshape(-1)).reshape(target_shape) for jj in ii] for ii in source_kernel.transpose([3, 2, 0, 1])])
        target_weights = target_weights.transpose([2, 3, 1, 0])
        self.kernel.assign(target_weights)
        self.bias.assign(source_bias)


def Beit(
    depth=12,
    embed_dim=768,
    num_heads=12,
    mlp_ratio=4,
    patch_size=16,
    attn_key_dim=0,
    attn_qv_bias=True,  # Default False for Vit, True for Beit, if True and attn_qkv_bias being False, will add BiasLayer for query and key.
    attn_qkv_bias=False,  # True for Vit, False for Beit, if True, will just use bias in qkv_dense, and set qv_bias False.
    attn_out_weight=True,
    attn_out_bias=True,
    attn_dropout=0,
    gamma_init_value=0.1,  # 0 for Vit, 0.1 for Beit, if > 0 will use `layer_scale` on block output
    use_abs_pos_emb=False,  # True for Vit, False for Beit, whether use abcolute positional embedding or relative one in attention blocks
    use_abs_pos_emb_on_cls_token=True,  # False for FlexiViT, no_embed_class in timm. If use_abs_pos_emb is True, whether apply pos_emb on cls_token.
    use_mean_pooling=True,  # False for Vit, True for Beit, whether use use mean output or `class_token` output
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    pretrained=None,
    force_reload_mismatch=False,  # set True if patch_size changed, will force reloading pos_emb and stem_conv weights
    model_name="beit",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)

    """ forward_embeddings """
    # nn = conv2d_no_bias(inputs, embed_dim, patch_size, strides=patch_size, padding="valid", use_bias=True, name="stem_")
    nn = PatchConv2DWithResampleWeights(embed_dim, patch_size, strides=patch_size, padding="valid", use_bias=True, name="stem_conv")(inputs)
    patch_height = nn.shape[1]
    nn = keras.layers.Reshape([-1, nn.shape[-1]])(nn)

    if use_abs_pos_emb and use_abs_pos_emb_on_cls_token:  # EvaLarge and EvaGiant
        nn = ClassToken(name="cls_token")(nn)
        nn = PositionalEmbedding(input_height=patch_height, name="positional_embedding")(nn)
    elif use_abs_pos_emb:  # FlexiViT models
        nn = PositionalEmbedding(input_height=patch_height, name="positional_embedding")(nn)
        nn = ClassToken(name="cls_token")(nn)
    else:  # Beit and BeitV2
        nn = ClassToken(name="cls_token")(nn)

    attn_params = {
        "num_heads": num_heads,
        "key_dim": attn_key_dim,
        "qv_bias": attn_qv_bias,
        "qkv_bias": attn_qkv_bias,
        "out_weight": attn_out_weight,
        "out_bias": attn_out_bias,
        "attn_height": patch_height,
        "attn_dropout": attn_dropout,
        "use_pos_emb": not use_abs_pos_emb,
    }

    """ forward_tokens """
    drop_connect_rates = drop_connect_rates_split([depth], 0.0, drop_connect_rate)[0]
    for id in range(depth):
        name = "block{}_".format(id)
        block_drop_rate = drop_connect_rates[id]
        nn = attention_mlp_block(nn, embed_dim, gamma_init_value, mlp_ratio, block_drop_rate, activation, attn_params, name=name)

    if use_mean_pooling:
        nn = tf.reduce_mean(nn[:, 1:, :], axis=1)
        nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="out_")
    else:
        nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="out_")[:, 0]

    if num_classes > 0:
        head_init = HeadInitializer()
        nn = keras.layers.Dense(
            num_classes, dtype="float32", activation=classifier_activation, kernel_initializer=head_init, bias_initializer=head_init, name="predictions"
        )(nn)
    model = tf.keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="tf")
    mismatch_class = [PatchConv2DWithResampleWeights, PositionalEmbedding if use_abs_pos_emb else MultiHeadRelativePositionalEmbedding]
    reload_model_weights(model, PRETRAINED_DICT, "beit", pretrained, mismatch_class, force_reload_mismatch)
    return model


def BeitBasePatch16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    embed_dim = 768
    depth = 12
    num_heads = 12
    gamma_init_value = 0.1
    force_reload_mismatch = kwargs.get("patch_size", 16) != 16  # If patch_size not 16, force reload pos_emb and stem_conv weights
    kwargs.pop("kwargs", None)  # From BeitV2BasePatch16
    return Beit(**locals(), model_name=kwargs.pop("model_name", "beit_base_patch16"), **kwargs)


def BeitV2BasePatch16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return BeitBasePatch16(**locals(), **kwargs, model_name="beit_v2_base_patch16")


def BeitLargePatch16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    embed_dim = 1024
    depth = 24
    num_heads = 16
    gamma_init_value = 1e-5
    force_reload_mismatch = kwargs.get("patch_size", 16) != 16  # If patch_size not 16, force reload pos_emb and stem_conv weights
    kwargs.pop("kwargs", None)  # From BeitV2LargePatch16
    return Beit(**locals(), model_name=kwargs.pop("model_name", "beit_large_patch16"), **kwargs)


def BeitV2LargePatch16(
    input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs
):
    return BeitLargePatch16(**locals(), **kwargs, model_name="beit_v2_large_patch16")


""" keras_model_load_weights_from_pytorch_model """


def keras_model_load_weights_from_pytorch_model(keras_model, timm_vit_model, save_name=None):
    from keras_cv_attention_models import download_and_load, attention_layers

    skip_weights = ["relative_position_index"]
    unstack_weights = ["cls_token", "gamma_1", "gamma_2", "relative_position_bias_table", "q_bias", "v_bias", "pos_embed"]
    tail_align_dict = {"attn_gamma": -6, "mlp_gamma": -9, "attn_query_bias": -1, "attn_value_bias": -1, "attn_pos_emb": -1}
    full_name_align_dict = {"cls_token": -2 if "flexivit" in keras_model.name else -1, "positional_embedding": -1}
    additional_transfer = {attention_layers.MultiHeadRelativePositionalEmbedding: lambda ww: [ww[0].T]}

    download_and_load.keras_reload_from_torch_model(
        torch_model=timm_vit_model,
        keras_model=keras_model,
        input_shape=keras_model.input_shape[1:-1],
        skip_weights=skip_weights,
        unstack_weights=unstack_weights,
        tail_align_dict=tail_align_dict,
        full_name_align_dict=full_name_align_dict,
        tail_split_position=1,
        additional_transfer=additional_transfer,
        save_name=save_name if save_name is not None else (keras_model.name + "_{}.h5".format(keras_model.input_shape[1])),
        do_convert=True,
        # do_predict=False if "eva_giant" in keras_model.name else True,
    )
