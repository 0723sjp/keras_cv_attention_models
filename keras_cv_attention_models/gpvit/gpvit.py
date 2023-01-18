import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    add_with_layer_scale_and_drop_block,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    layer_norm,
    mlp_block,
    mlp_mixer_block,
    output_block,
    PositionalEmbedding,
    scaled_dot_product_attention,
    window_attention,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "gpvit_l1": {"imagenet": {224: "4c2b124acdb20cc9ef33a32a85a2cd4e"}},
    "gpvit_l2": {"imagenet": {224: "ce5e72b80cfcb9a9567e8e52d42b4e15"}},
    "gpvit_l3": {"imagenet": {224: "a378001b60878ea8851ebe78a28bcfbe"}},
    "gpvit_l4": {"imagenet": {224: "0cde8fcea39794ea0ce1ffdf7c49eef0"}},
}


@tf.keras.utils.register_keras_serializable(package="kecam")
class PureWeigths(keras.layers.Layer):
    """Just return a weights with specific shape"""

    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def build(self, input_shape):
        self.gain = self.add_weight(name="gain", shape=self.shape, dtype=self.dtype, trainable=True)

    def call(self, inputs, **kwargs):
        return self.gain

    def get_config(self):
        config = super().get_config()
        config.update({"shape": self.shape})
        return config


def lepe_attention(inputs, num_heads=4, dropout=0, name=""):
    query, key, value = tf.split(inputs, 3, axis=-1)
    _, hh, ww, input_channel = query.shape
    blocks = hh * ww
    key_dim = input_channel // num_heads
    # print(f"{input_channel = }, {num_heads = }, {key_dim = }")

    lepe = depthwise_conv2d_no_bias(value, kernel_size=3, use_bias=True, padding="same", name=name + "lepe_")

    query = tf.transpose(tf.reshape(query, [-1, blocks, num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dim]
    key = tf.transpose(tf.reshape(key, [-1, blocks, num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = tf.transpose(tf.reshape(value, [-1, blocks, num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dim]
    # print(f"{query.shape = }, {key.shape = }, {value.shape = }, {lepe.shape = }")

    output_shape = (hh, ww, input_channel)
    attn_out = scaled_dot_product_attention(query, key, value, output_shape=output_shape, out_weight=False, dropout=dropout, name=name)
    return keras.layers.Add()([attn_out, lepe])


def window_lepe_attention(inputs, num_heads=4, window_size=2, key_dim=0, qkv_bias=True, dropout=0, name=""):
    _, hh, ww, input_channel = inputs.shape
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    # out_shape = input_channel if out_shape is None or not out_weight else out_shape
    emb_dim = num_heads * key_dim

    # qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3).contiguous()
    # split 3 -> split 2
    qkv = keras.layers.Dense(emb_dim * 3, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    qkv = tf.reshape(qkv, [-1, hh * ww * 3, emb_dim])
    qkv_left, qkv_right = tf.split(qkv, 2, axis=-1)
    qkv_left = tf.reshape(qkv_left, [-1, hh, ww, 3 * qkv_left.shape[-1]])
    qkv_right = tf.reshape(qkv_right, [-1, hh, ww, 3 * qkv_right.shape[-1]])

    attn_out_left = window_attention(qkv_left, window_size=(hh, window_size), num_heads=num_heads // 2, attention_block=lepe_attention, name=name + "left_")
    attn_out_right = window_attention(qkv_right, window_size=(window_size, ww), num_heads=num_heads // 2, attention_block=lepe_attention, name=name + "right_")
    attn_out = tf.concat([attn_out_left, attn_out_right], axis=-1)
    attn_out = keras.layers.Dense(attn_out.shape[-1], use_bias=True, name=name and name + "attn_out")(attn_out)
    return attn_out


def window_lepe_attention_mlp_block(inputs, num_heads=4, window_size=2, mlp_ratio=4, layer_scale=0.1, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1]

    """ attention """
    nn = layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, name=name + "attn_")
    nn = window_lepe_attention(nn, num_heads=num_heads, window_size=window_size, name=name + "attn_")
    attn_out = add_with_layer_scale_and_drop_block(inputs, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "attn_")

    """ MLP """
    nn = layer_norm(attn_out, epsilon=LAYER_NORM_EPSILON, name=name + "mlp_")
    nn = mlp_block(nn, input_channel * mlp_ratio, activation=activation, name=name + "mlp_")
    nn = add_with_layer_scale_and_drop_block(attn_out, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "mlp_")

    """ DepthWise """
    nn = depthwise_conv2d_no_bias(nn, kernel_size=3, padding="same", name=name + "output_")
    return nn


def light_group_attention(inputs, num_heads=4, key_dim=0, num_group_token=0, use_key_value_norm=True, qkv_bias=True, dropout=0, name=""):
    _, hh, ww, input_channel = inputs.shape
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    blocks = hh * ww

    query = PureWeigths(shape=[1, num_group_token, input_channel], name=name + "query")(inputs)
    query = layer_norm(query, epsilon=LAYER_NORM_EPSILON, name=name + "query_")
    key_value = layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, name=name + "key_value_") if use_key_value_norm else inputs
    key = keras.layers.Dense(key_value.shape[-1], use_bias=qkv_bias, name=name and name + "key")(key_value)

    query = tf.transpose(tf.reshape(query, [-1, num_group_token, num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dim]
    key = tf.transpose(tf.reshape(key, [-1, blocks, num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = tf.transpose(tf.reshape(key_value, [-1, blocks, num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dim]

    output_shape = [num_group_token, input_channel]
    return scaled_dot_product_attention(query, key, value, output_shape=output_shape, out_weight=False, dropout=dropout, name=name)


def full_ungroup_attention(inputs, group_token, num_heads=4, key_dim=0, qkv_bias=True, dropout=0, name=""):
    # inputs: [batch, height, width, channel], group_token: [batch, num_group_token, channel]
    input_channel = inputs.shape[-1]
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads

    query = layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, name=name + "query_")
    key_value = layer_norm(group_token, epsilon=LAYER_NORM_EPSILON, name=name + "key_value_")

    query = keras.layers.Dense(query.shape[-1], use_bias=qkv_bias, name=name and name + "query")(query)
    key = keras.layers.Dense(key_value.shape[-1], use_bias=qkv_bias, name=name and name + "key")(key_value)
    value = keras.layers.Dense(key_value.shape[-1], use_bias=qkv_bias, name=name and name + "value")(key_value)

    query = tf.transpose(tf.reshape(query, [-1, query.shape[1] * query.shape[2], num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dim]
    key = tf.transpose(tf.reshape(key, [-1, key_value.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = tf.transpose(tf.reshape(value, [-1, key_value.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dim]

    attn = scaled_dot_product_attention(query, key, value, output_shape=inputs.shape, out_weight=True, out_bias=True, dropout=dropout, name=name)

    attn_out = tf.concat([inputs, attn], axis=-1)
    attn_out = keras.layers.Dense(inputs.shape[-1], use_bias=True, name=name and name + "attn_out")(attn_out)
    return attn_out


def group_attention(inputs, num_heads=4, num_group_token=0, mlp_ratio=4, layer_scale=0.1, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1]
    group_token = light_group_attention(inputs, num_heads=num_heads, num_group_token=num_group_token, name=name + "light_attn_")

    tokens_mlp_dim, channels_mlp_dim = input_channel * 0.5, input_channel * 4  # all using embed_dims
    group_token = mlp_mixer_block(group_token, tokens_mlp_dim, channels_mlp_dim, drop_rate=drop_rate, activation=activation, name=name)

    attn_out = full_ungroup_attention(inputs, group_token, num_heads=num_heads, name=name + "full_attn_")
    attn_out = keras.layers.Reshape(inputs.shape[1:])(attn_out)

    """ MLP """
    nn = layer_norm(attn_out, epsilon=LAYER_NORM_EPSILON, name=name + "mlp_")
    nn = mlp_block(nn, input_channel * mlp_ratio, activation=activation, name=name + "mlp_")
    nn = add_with_layer_scale_and_drop_block(attn_out, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "mlp_")

    """ DepthWise """
    nn = depthwise_conv2d_no_bias(nn, kernel_size=3, padding="same", name=name)
    nn = batchnorm_with_activation(nn, activation="relu", name=name + "output_")
    return nn


def GPViT(
    num_layers=12,
    embed_dims=216,
    stem_depth=1,
    num_window_heads=12,
    num_group_heads=6,
    mlp_ratios=4,
    window_size=2,
    group_attention_layer_ids=[1, 4, 7, 10],
    group_attention_layer_group_tokens=[64, 32, 32, 16],
    use_neck_attention_output=True,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    dropout=0,
    layer_scale=0,
    classifier_activation="softmax",
    pretrained=None,
    model_name="gp_vit",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)

    """ Stem """
    nn = conv2d_no_bias(inputs, 64, kernel_size=7, strides=2, padding="same", name="stem_")
    nn = batchnorm_with_activation(nn, activation="relu", name="stem_")
    for id in range(stem_depth - 1):
        nn = conv2d_no_bias(nn, 64, kernel_size=3, strides=1, padding="same", name="stem_{}_".format(id + 1))
        nn = batchnorm_with_activation(nn, activation="relu", name="stem_{}_".format(id + 1))
    ## nn = AdaptivePadding(nn) with padding='corner' --> use_torch_padding=False
    nn = conv2d_no_bias(nn, embed_dims, kernel_size=4, strides=4, padding="same", use_torch_padding=False, use_bias=True, name="stem_patch_")
    nn = PositionalEmbedding(name="positional_embedding")(nn)

    """ stacks """
    group_attention_layer_group_tokens = group_attention_layer_group_tokens.copy()
    global_block_id = 0
    for block_id in range(num_layers):
        block_name = "block{}_".format(block_id + 1)
        block_drop_rate = drop_connect_rate * global_block_id / num_layers
        if block_id in group_attention_layer_ids:
            num_group_token = group_attention_layer_group_tokens.pop(0)
            nn = group_attention(nn, num_group_heads, num_group_token, mlp_ratios, layer_scale, block_drop_rate, activation=activation, name=block_name)
        else:
            nn = window_lepe_attention_mlp_block(nn, num_window_heads, window_size, mlp_ratios, layer_scale, block_drop_rate, activation, name=block_name)
        global_block_id += 1

    nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="pre_out_")
    if use_neck_attention_output:
        nn = light_group_attention(nn, num_heads=6, num_group_token=1, use_key_value_norm=False, qkv_bias=False, name="neck_")
        nn = tf.squeeze(nn, axis=1)

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = tf.keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "gpvit", pretrained, PositionalEmbedding)
    return model


def GPViT_L1(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return GPViT(**locals(), model_name="gpvit_l1", **kwargs)


def GPViT_L2(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dims = 348
    stem_depth = 2
    return GPViT(**locals(), model_name="gpvit_l2", **kwargs)


def GPViT_L3(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dims = 432
    stem_depth = 2
    use_neck_attention_output = False
    return GPViT(**locals(), model_name="gpvit_l3", **kwargs)


def GPViT_L4(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dims = 624
    stem_depth = 3
    use_neck_attention_output = False
    return GPViT(**locals(), model_name="gpvit_l4", **kwargs)
