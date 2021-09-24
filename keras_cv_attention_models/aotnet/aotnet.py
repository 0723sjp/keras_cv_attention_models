import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import os
from keras_cv_attention_models.attention_layers import anti_alias_downsample, batchnorm_with_activation, conv2d_no_bias, drop_block, se_module
from keras_cv_attention_models import attention_layers


MHSA_PARAMS = {"num_heads": 4, "relative": True, "out_bias": True}
HALO_PARAMS = {"block_size": 4, "halo_size": 1, "num_heads": 8, "out_bias": True}
SA_PARAMS = {"kernel_size": 3, "groups": 2}
COT_PARAMS = {"kernel_size": 3}
OUTLOOK_PARAMS = {"num_heads": 6, "kernel_size": 3}
GROUPS_CONV_PARAMS = {"groups": 32, "kernel_size": 3}


def attn_block(inputs, filters, strides=1, attn_type=None, se_ratio=0, use_bn=True, activation="relu", name=""):
    nn = inputs
    if attn_type == "mhsa":  # MHSA block
        nn = attention_layers.mhsa_with_relative_position_embedding(nn, **MHSA_PARAMS, name=name + "_mhsa_")
    elif attn_type == "halo":  # HaloAttention
        key_dim = filters // num_heads
        nn = attention_layers.HaloAttention(**HALO_PARAMS, key_dim=key_dim, strides=strides, name=name + "halo")(nn)
    elif attn_type == "sa":  # split_attention_conv2d
        nn = attention_layers.split_attention_conv2d(nn, **SA_PARAMS, filters=filters, strides=strides, activation=activation, name=name + "sa_")
    elif attn_type == "cot":  # cot_attention
        nn = attention_layers.cot_attention(nn, **COT_PARAMS, activation=activation, name=name + "cot_")
    elif attn_type == "outlook":  # outlook_attention
        nn = attention_layers.outlook_attention(nn, filters, **OUTLOOK_PARAMS, name=name + "outlook_")
    elif attn_type == "groups_conv":  # ResNeXt like
        nn = conv2d_no_bias(nn, filters, **GROUPS_CONV_PARAMS, strides=strides, padding="SAME", name=name + "GC_")
    else:  # ResNet block
        nn = conv2d_no_bias(nn, filters, 3, strides=strides, padding="SAME", name=name + "conv_")

    if attn_type in ["mhsa", "cot", "outlook"] and strides != 1:  # Downsample
        nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = keras.layers.AveragePooling2D(pool_size=3, strides=strides, name=name + "pool")(nn)

    if use_bn:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name)

    if se_ratio:
        nn = se_module(nn, se_ratio=se_ratio, activation=activation, name=name + "se_")
    return nn


def conv_shortcut_branch(inputs, expanded_filter, preact=False, strides=1, avg_pool_down=False, anti_alias_down=False, name=""):
    if strides > 1 and avg_pool_down:
        shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shorcut_down")(inputs)
        strides = 1
    elif strides > 1 and anti_alias_down:
        shortcut = anti_alias_downsample(inputs, kernel_size=3, strides=2, name=name + "shorcut_down")
        strides = 1
    else:
        shortcut = inputs
    shortcut = conv2d_no_bias(shortcut, expanded_filter, 1, strides=strides, name=name + "shortcut_")
    if not preact:  # ResNet
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shortcut_")
    return shortcut


def deep_branch(inputs, filters, strides=1, expansion=4, attn_type=None, se_ratio=0, use_3x3_kernel=False, activation="relu", name=""):
    expanded_filter = filters * expansion
    if use_3x3_kernel:
        nn = conv2d_no_bias(inputs, filters, 3, strides=1, padding="SAME", name=name + "deep_1_")  # Using strides=1 for not changing input shape
        # nn = conv2d_no_bias(inputs, filters, 3, strides=strides, padding="SAME", name=name + "1_")
        # strides = 1
    else:
        nn = conv2d_no_bias(inputs, filters, 1, strides=1, padding="VALID", name=name + "deep_1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "deep_1_")
    use_bn = not use_3x3_kernel
    nn = attn_block(nn, filters, strides, attn_type, se_ratio / expansion, use_bn, activation, name=name + "deep_2_")

    if not use_3x3_kernel:
        nn = conv2d_no_bias(nn, expanded_filter, 1, strides=1, padding="VALID", name=name + "deep_3_")
    return nn


def block(
    inputs,
    filters,
    strides=1,
    conv_shortcut=False,
    expansion=4,
    attn_type=None,
    se_ratio=0,
    drop_rate=0,
    preact=False,
    use_3x3_kernel=False,
    avg_pool_down=False,
    anti_alias_down=False,
    activation="relu",
    name="",
):
    expanded_filter = filters * expansion
    halo_block_size = HALO_PARAMS["block_size"]
    if attn_type == "halo" and inputs.shape[1] % halo_block_size != 0:  # HaloAttention
        gap = halo_block_size - inputs.shape[1] % halo_block_size
        pad_head, pad_tail = gap // 2, gap - gap // 2
        inputs = keras.layers.ZeroPadding2D(padding=((pad_head, pad_tail), (pad_head, pad_tail)), name=name + "gap_pad")(inputs)

    if preact:  # ResNetV2
        pre_inputs = batchnorm_with_activation(inputs, activation=activation, zero_gamma=False, name=name + "preact_")
    else:
        pre_inputs = inputs

    if conv_shortcut:  # Set a new shortcut using conv
        shortcut = conv_shortcut_branch(pre_inputs, expanded_filter, preact, strides, avg_pool_down, anti_alias_down, name=name)
    else:
        shortcut = keras.layers.MaxPooling2D(strides, strides=strides, padding="SAME")(inputs) if strides > 1 else inputs
    deep = deep_branch(pre_inputs, filters, strides, expansion, attn_type, se_ratio, use_3x3_kernel, activation=activation, name=name)

    # print(">>>> shortcut:", shortcut.shape, "deep:", deep.shape)
    if preact:  # ResNetV2
        deep = drop_block(deep, drop_rate)
        return keras.layers.Add(name=name + "add")([shortcut, deep])
    else:
        deep = batchnorm_with_activation(deep, activation=None, zero_gamma=True, name=name + "3_")
        deep = drop_block(deep, drop_rate)
        out = keras.layers.Add(name=name + "add")([shortcut, deep])
        return keras.layers.Activation(activation, name=name + "out")(out)


def stack(inputs, blocks, filters, strides=2, strides_first=True, expansion=4, attn_types=None, se_ratio=0, stack_drop=0, block_params={}, name=""):
    nn = inputs
    # print(">>>> attn_types:", attn_types)
    strides_block_id = 0 if strides_first else blocks - 1
    for id in range(blocks):
        conv_shortcut = True if id == 0 and (strides != 1 or inputs.shape[-1] != filters * expansion) else False
        cur_strides = strides if id == strides_block_id else 1
        block_name = name + "block{}_".format(id + 1)
        block_drop_rate = stack_drop[id] if isinstance(stack_drop, (list, tuple)) else stack_drop
        attn_type = attn_types[id] if isinstance(attn_types, (list, tuple)) else attn_types
        cur_se_ratio = se_ratio[id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        nn = block(nn, filters, cur_strides, conv_shortcut, expansion, attn_type, cur_se_ratio, block_drop_rate, **block_params, name=block_name)
    return nn


def aot_stem(inputs, stem_width, activation="relu", deep_stem=False, quad_stem=False, quad_stem_act=False, name=""):
    if deep_stem:
        nn = conv2d_no_bias(inputs, stem_width // 2, 3, strides=2, padding="same", name=name + "1_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
        nn = conv2d_no_bias(nn, stem_width // 2, 3, strides=1, padding="same", name=name + "2_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "2_")
        nn = conv2d_no_bias(nn, stem_width, 3, strides=1, padding="same", name=name + "3_")
    elif quad_stem:
        nn = conv2d_no_bias(inputs, stem_width // 8, 3, strides=2, padding="same", name=name + "1_")
        if quad_stem_act:
            nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
        nn = conv2d_no_bias(nn, stem_width // 4, 3, strides=1, padding="same", name=name + "2_")
        if quad_stem_act:
            nn = batchnorm_with_activation(nn, activation=activation, name=name + "2_")
        nn = conv2d_no_bias(nn, stem_width // 2, 3, strides=1, padding="same", name=name + "3_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "3_")
        nn = conv2d_no_bias(nn, stem_width, 3, strides=2, padding="same", name=name + "4_")
    else:
        nn = conv2d_no_bias(inputs, stem_width, 7, strides=2, padding="same", name=name)
    return nn


def AotNet(
    num_blocks,
    preact=False,
    strides=[1, 2, 2, 1],
    strides_first=True,  # True for resnet, False for resnetv2
    out_channels=[64, 128, 256, 512],
    expansion=4,
    stem_width=64,
    deep_stem=False,
    quad_stem=False,
    stem_downsample=True,
    attn_types=None,
    se_ratio=0,  # (0, 1)
    num_features=0,
    use_3x3_kernel=False,
    avg_pool_down=False,
    anti_alias_down=False,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    drop_rate=0,
    model_name="aotnet",
    kwargs=None,
):
    inputs = keras.layers.Input(shape=input_shape)
    nn = aot_stem(inputs, stem_width, activation=activation, deep_stem=deep_stem, quad_stem=quad_stem, name="stem_")

    if not preact:
        nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    if stem_downsample:
        nn = keras.layers.ZeroPadding2D(padding=1, name="stem_pool_pad")(nn)
        nn = keras.layers.MaxPooling2D(pool_size=3, strides=2, name="stem_pool")(nn)

    block_params = {  # params same for all blocks
        "preact": preact,
        "use_3x3_kernel": use_3x3_kernel,
        "avg_pool_down": avg_pool_down,
        "anti_alias_down": anti_alias_down,
        "activation": activation,
    }

    drop_connect_rates = tf.split(tf.linspace(0., drop_connect_rate, sum(num_blocks)), num_blocks)
    drop_connect_rates = [ii.numpy().tolist() for ii in drop_connect_rates]
    for id, (num_block, out_channel, stride, drop_connect) in enumerate(zip(num_blocks, out_channels, strides, drop_connect_rates)):
        name = "stack{}_".format(id + 1)
        attn_type = attn_types[id] if isinstance(attn_types, (list, tuple)) else attn_types
        cur_se_ratio = se_ratio[id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        cur_expansion = expansion[id] if isinstance(expansion, (list, tuple)) else expansion
        nn = stack(nn, num_block, out_channel, stride, strides_first, cur_expansion, attn_type, cur_se_ratio, drop_connect, block_params, name=name)

    if preact:  # resnetv2 like
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name="post_")

    if num_features != 0:  # efficientnet like
        nn = conv2d_no_bias(nn, num_features, 1, strides=1, name="features_")
        nn = batchnorm_with_activation(nn, activation=activation, name="features_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model


def AotNet50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 6, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet50", **kwargs)


def AotNet101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 23, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet101", **kwargs)


def AotNet152(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 8, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet152", **kwargs)


def AotNet200(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 24, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet200", **kwargs)


def AotNetV2(num_blocks, preact=True, strides=1, strides_first=False, **kwargs):
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNet(num_blocks, preact=preact, strides=strides, strides_first=strides_first, **kwargs)


def AotNet50V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 6, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name="aotnet50v2", **kwargs)


def AotNet101V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 23, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name="aotnet101v2", **kwargs)


def AotNet152V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 8, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name="aotnet152v2", **kwargs)


def AotNet200V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 24, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name="aotnet200v2", **kwargs)
