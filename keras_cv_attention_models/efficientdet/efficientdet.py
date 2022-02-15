import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models import model_surgery
from keras_cv_attention_models import efficientnet
from keras_cv_attention_models.attention_layers import activation_by_name, add_pre_post_process
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.coco.eval_func import DecodePredictions

BATCH_NORM_EPSILON = 1e-3
PRETRAINED_DICT = {
    "efficientdet_d0": {"coco": {512: "d5ba1c9ffd3627571bb00091e3fde28e"}},
    "efficientdet_d1": {"coco": {640: "c34d8603ea1eefbd7f10415a5b51d824"}},
    "efficientdet_d2": {"coco": {768: "bfd7d19cd5a770e478b51c09e2182dc7"}},
    "efficientdet_d3": {"coco": {896: "39d7b5ac515e073123686c200573b7be"}},
    "efficientdet_d4": {"coco": {1024: "b80d45757f139ef27bf5cf8de0746222"}},
    "efficientdet_d5": {"coco": {1280: "e7bc8987e725d8779684da580a42efa9"}},
    "efficientdet_d6": {"coco": {1280: "9302dc59b8ade929ea68bd68fc71c9f3"}},
    "efficientdet_d7": {"coco": {1536: "6d615faf95a4891aec12392f17155950"}},
    "efficientdet_d7x": {"coco": {1536: "a1e4b4e8e488eeabee14c7801028984d"}},
    "efficientdet_lite0": {"coco": {320: "f6d9aeb36eb0377ff135b9d367f8b4a6"}},
    "efficientdet_lite1": {"coco": {384: "7a593f798462e558deeec6ef976574fc"}},
    "efficientdet_lite2": {"coco": {448: "b48fa6246a50d3d59785cc119c52fc94"}},
    "efficientdet_lite3": {"coco": {512: "945e66f31622d2806aeafde4dff3b4b7"}},
    "efficientdet_lite3x": {"coco": {640: "bcd55e31b99616439ad4f988e2337b86"}},
    "efficientdet_lite4": {"coco": {640: "9f5eadec38498faffb8d6777cb09e3a7"}},
}


@tf.keras.utils.register_keras_serializable(package="efficientdet")
class ReluWeightedSum(keras.layers.Layer):
    def __init__(self, initializer="ones", epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.initializer, self.epsilon = initializer, epsilon

    def build(self, input_shape):
        self.total = len(input_shape)
        self.gain = self.add_weight(name="gain", shape=(self.total,), initializer=self.initializer, dtype=self.dtype, trainable=True)
        self.__epsilon__ = tf.cast(self.epsilon, self._compute_dtype)

    def call(self, inputs):
        gain = tf.nn.relu(self.gain)
        gain = gain / (tf.reduce_sum(gain) + self.__epsilon__)
        return tf.reduce_sum([inputs[id] * gain[id] for id in range(self.total)], axis=0)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"initializer": self.initializer, "epsilon": self.epsilon})
        return base_config


def align_feature_channel(inputs, output_channel, name=""):
    # print(f">>>> align_feature_channel: {name = }, {inputs.shape = }, {output_channel = }")
    nn = inputs
    if inputs.shape[-1] != output_channel:
        nn = keras.layers.Conv2D(output_channel, kernel_size=1, name=name + "channel_conv")(nn)
        nn = keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, name=name + "channel_bn")(nn)
    return nn


def resample_fuse(inputs, output_channel, use_weighted_sum=True, interpolation="nearest", use_sep_conv=True, activation="swish", name=""):
    inputs[0] = align_feature_channel(inputs[0], output_channel, name=name)

    if use_weighted_sum:
        nn = ReluWeightedSum(name=name + "wsm")(inputs)
    else:
        nn = keras.layers.Add(name=name + "sum")(inputs)
    nn = activation_by_name(nn, activation, name=name)
    if use_sep_conv:
        nn = keras.layers.SeparableConv2D(output_channel, kernel_size=3, padding="SAME", use_bias=True, name=name + "sepconv")(nn)
    else:
        nn = keras.layers.Conv2D(output_channel, kernel_size=3, padding="SAME", use_bias=True, name=name + "conv")(nn)
    nn = keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, name=name + "bn")(nn)
    return nn


def bi_fpn(features, output_channel, use_weighted_sum=True, use_sep_conv=True, interpolation="nearest", activation="swish", name=""):
    # print(f">>>> bi_fpn: {[ii.shape for ii in features] = }")
    # features: [p3, p4, p5, p6, p7]
    up_features = [features[-1]]
    for id, feature in enumerate(features[:-1][::-1]):
        cur_name = name + "p{}_up_".format(len(features) - id + 1)
        # up_feature = keras.layers.UpSampling2D(size=(2, 2), interpolation=interpolation, name=cur_name + "up")(up_features[-1])
        up_feature = tf.image.resize(up_features[-1], tf.shape(feature)[1:-1], method=interpolation)
        up_feature = resample_fuse([feature, up_feature], output_channel, use_weighted_sum, use_sep_conv=use_sep_conv, activation=activation, name=cur_name)
        up_features.append(up_feature)

    # up_features: [p7, p6_up, p5_up, p4_up, p3_up]
    out_features = [up_features[-1]]  # [p3_up]
    up_features = up_features[1:-1][::-1]  # [p4_up, p5_up, p6_up]
    for id, feature in enumerate(features[1:]):
        cur_name = name + "p{}_out_".format(len(features) - 1 + id)
        down_feature = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="SAME", name=cur_name + "max_down")(out_features[-1])
        fusion_feature = [feature, down_feature] if id == len(up_features) else [feature, up_features[id], down_feature]
        out_feature = resample_fuse(fusion_feature, output_channel, use_weighted_sum, use_sep_conv=use_sep_conv, activation=activation, name=cur_name)
        out_features.append(out_feature)

    # out_features: [p3_up, p4_out, p5_out, p6_out, p7_out]
    return out_features


def det_head(features, filters, depth, classes=80, anchors=9, bias_init="zeros", use_sep_conv=True, activation="swish", head_activation="sigmoid", name=""):
    if use_sep_conv:
        names = [name + "{}_sepconv".format(id + 1) for id in range(depth)]
        convs = [keras.layers.SeparableConv2D(filters, kernel_size=3, padding="SAME", use_bias=True, name=names[id]) for id in range(depth)]
    else:
        names = [name + "{}_conv".format(id + 1) for id in range(depth)]
        convs = [keras.layers.Conv2D(filters, kernel_size=3, padding="SAME", use_bias=True, name=names[id]) for id in range(depth)]

    outputs = []
    for feature_id, feature in enumerate(features):
        nn = feature
        for id in range(depth):
            nn = convs[id](nn)
            cur_name = name + "{}_{}_bn".format(id + 1, feature_id + 1)
            nn = keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON, name=cur_name)(nn)
            nn = activation_by_name(nn, activation, name=cur_name + "{}_".format(id + 1))
        outputs.append(nn)

    if classes > 0:
        if use_sep_conv:
            header_conv = keras.layers.SeparableConv2D(classes * anchors, kernel_size=3, padding="SAME", bias_initializer=bias_init, name=name + "head")
        else:
            header_conv = keras.layers.Conv2D(classes * anchors, kernel_size=3, padding="SAME", bias_initializer=bias_init, name=name + "conv_head")
        outputs = [header_conv(ii) for ii in outputs]
        outputs = [keras.layers.Reshape([-1, classes])(ii) for ii in outputs]
        outputs = tf.concat(outputs, axis=1)
        outputs = activation_by_name(outputs, head_activation, name=name + "output_")
    return outputs


def EfficientDet(
    backbone,
    features_pick=[-3, -2, -1],  # The last 3 ones, or specific layer_names / feature_indexes
    additional_features=2,  # Add p5->p6, p6->p7
    fpn_depth=3,
    head_depth=3,
    num_channels=64,
    use_weighted_sum=True,
    num_anchors=9,
    num_classes=90,
    use_sep_conv=True,
    activation="swish",
    classifier_activation="sigmoid",
    freeze_backbone=False,
    anchor_scale=4,  # Init anchors for model prediction
    pyramid_levels=[3, 7],  # Init anchors for model prediction
    pretrained="coco",
    model_name=None,
    rescale_mode="torch",
    kwargs=None,  # Not using, recieving parameter
    input_shape=None,  # Not using, recieving parameter
):
    if freeze_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True

    if isinstance(features_pick[0], str):
        fpn_features = [backbone.get_layer(layer_name) for layer_name in features_pick]
    else:
        features = model_surgery.get_pyramide_feture_layers(backbone)
        fpn_features = [features[id] for id in features_pick]
    print(">>>> features:", {ii.name: ii.output_shape for ii in fpn_features})
    fpn_features = [ii.output for ii in fpn_features]

    # Build additional input features that are not from backbone.
    for id in range(additional_features):
        cur_name = "p{}_p{}_".format(id + 5, id + 6)
        additional_feature = align_feature_channel(fpn_features[-1], num_channels, name=cur_name)
        additional_feature = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="SAME", name=cur_name + "max_down")(additional_feature)
        fpn_features.append(additional_feature)

    for id in range(fpn_depth):
        fpn_features = bi_fpn(fpn_features, num_channels, use_weighted_sum, use_sep_conv, activation=activation, name="biFPN_{}_".format(id + 1))

    # Save line-width
    head_kwargs = {"filters": num_channels, "depth": head_depth, "anchors": num_anchors, "use_sep_conv": use_sep_conv, "activation": activation}
    bbox_out = det_head(fpn_features, classes=4, bias_init="zeros", head_activation=None, name="regressor_", **head_kwargs)
    if num_classes > 0:
        bias_init = tf.constant_initializer(-tf.math.log((1 - 0.01) / 0.01).numpy())
        cls_out = det_head(fpn_features, classes=num_classes, bias_init=bias_init, head_activation=classifier_activation, name="classifier_", **head_kwargs)
        outputs = tf.concat([bbox_out, cls_out], axis=-1)
    else:
        outputs = bbox_out
    outputs = keras.layers.Activation("linear", dtype="float32", name="outputs_fp32")(outputs)

    model_name = model_name or backbone.name + "_det"
    model = keras.models.Model(inputs=backbone.inputs[0], outputs=outputs, name=model_name)
    add_pre_post_process(model, rescale_mode=rescale_mode, post_process=DecodePredictions(backbone.input_shape[1:], pyramid_levels, anchor_scale))
    reload_model_weights(model, PRETRAINED_DICT, "efficientdet", pretrained)
    # model.backbone = backbone
    return model


def EfficientDetD0(input_shape=(512, 512, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="swish", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1B0(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation, pretrained=None)
        features_pick = ["stack_2_block1_output", "stack_4_block2_output", "stack_6_block0_output"]
    model_name = kwargs.pop("model_name", "efficientdet_d0")
    return EfficientDet(**locals(), fpn_depth=3, head_depth=3, num_channels=64, **kwargs)


def EfficientDetD1(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="swish", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1B1(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation, pretrained=None)
        features_pick = ["stack_2_block2_output", "stack_4_block3_output", "stack_6_block1_output"]
    model_name = kwargs.pop("model_name", "efficientdet_d1")
    return EfficientDet(**locals(), fpn_depth=4, head_depth=3, num_channels=88, **kwargs)


def EfficientDetD2(input_shape=(768, 768, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="swish", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1B2(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation, pretrained=None)
        features_pick = ["stack_2_block2_output", "stack_4_block3_output", "stack_6_block1_output"]
    model_name = kwargs.pop("model_name", "efficientdet_d2")
    return EfficientDet(**locals(), fpn_depth=5, head_depth=3, num_channels=112, **kwargs)


def EfficientDetD3(input_shape=(896, 896, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="swish", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1B3(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation, pretrained=None)
        features_pick = ["stack_2_block2_output", "stack_4_block4_output", "stack_6_block1_output"]
    model_name = kwargs.pop("model_name", "efficientdet_d3")
    return EfficientDet(**locals(), fpn_depth=6, head_depth=4, num_channels=160, **kwargs)


def EfficientDetD4(input_shape=(1024, 1024, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="swish", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1B4(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation, pretrained=None)
        features_pick = ["stack_2_block3_output", "stack_4_block5_output", "stack_6_block1_output"]
    model_name = kwargs.pop("model_name", "efficientdet_d4")
    return EfficientDet(**locals(), fpn_depth=7, head_depth=4, num_channels=224, **kwargs)


def EfficientDetD5(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="swish", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1B5(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation, pretrained=None)
        features_pick = ["stack_2_block4_output", "stack_4_block6_output", "stack_6_block2_output"]
    model_name = kwargs.pop("model_name", "efficientdet_d5")
    return EfficientDet(**locals(), fpn_depth=7, head_depth=4, num_channels=288, **kwargs)


def EfficientDetD6(input_shape=(1280, 1280, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="swish", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1B6(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation, pretrained=None)
        features_pick = ["stack_2_block5_output", "stack_4_block7_output", "stack_6_block2_output"]
    model_name = kwargs.pop("model_name", "efficientdet_d6")
    return EfficientDet(**locals(), fpn_depth=8, head_depth=5, num_channels=384, use_weighted_sum=False, **kwargs)


def EfficientDetD7(input_shape=(1536, 1536, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="swish", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1B6(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation, pretrained=None)
        features_pick = ["stack_2_block5_output", "stack_4_block7_output", "stack_6_block2_output"]
    anchor_scale = kwargs.pop("anchor_scale", 5)
    model_name = kwargs.pop("model_name", "efficientdet_d7")
    return EfficientDet(**locals(), fpn_depth=8, head_depth=5, num_channels=384, use_weighted_sum=False, **kwargs)


def EfficientDetD7X(input_shape=(1536, 1536, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="swish", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1B7(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation, pretrained=None)
        features_pick = ["stack_2_block6_output", "stack_4_block9_output", "stack_6_block3_output"]
    pyramid_levels = kwargs.pop("pyramid_levels", [3, 8])
    additional_features = max(pyramid_levels) - 5  # 8 -> 3, 7 -> 2
    model_name = kwargs.pop("model_name", "efficientdet_d7x")
    return EfficientDet(**locals(), fpn_depth=8, head_depth=5, num_channels=384, use_weighted_sum=False, **kwargs)


""" EfficientDetLite models """


def EfficientDetLite0(input_shape=(320, 320, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="relu6", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1Lite0(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation)
        features_pick = ["stack_2_block1_output", "stack_4_block2_output", "stack_6_block0_output"]
    use_weighted_sum = False
    return EfficientDet(**locals(), fpn_depth=3, head_depth=3, num_channels=64, anchor_scale=3, rescale_mode="tf", model_name="efficientdet_lite0", **kwargs)


def EfficientDetLite1(input_shape=(384, 384, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="relu6", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1Lite1(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation)
        features_pick = ["stack_2_block2_output", "stack_4_block3_output", "stack_6_block0_output"]
    use_weighted_sum = False
    return EfficientDet(**locals(), fpn_depth=4, head_depth=3, num_channels=88, anchor_scale=3, rescale_mode="tf", model_name="efficientdet_lite1", **kwargs)


def EfficientDetLite2(input_shape=(448, 448, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="relu6", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1Lite2(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation)
        features_pick = ["stack_2_block2_output", "stack_4_block3_output", "stack_6_block0_output"]
    use_weighted_sum = False
    return EfficientDet(**locals(), fpn_depth=5, head_depth=3, num_channels=112, anchor_scale=3, rescale_mode="tf", model_name="efficientdet_lite2", **kwargs)


def EfficientDetLite3(input_shape=(512, 512, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="relu6", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1Lite3(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation)
        features_pick = ["stack_2_block2_output", "stack_4_block4_output", "stack_6_block0_output"]
    use_weighted_sum = False
    return EfficientDet(**locals(), fpn_depth=6, head_depth=4, num_channels=160, anchor_scale=4, rescale_mode="tf", model_name="efficientdet_lite3", **kwargs)


def EfficientDetLite3X(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="relu6", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1Lite3(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation)
        features_pick = ["stack_2_block2_output", "stack_4_block4_output", "stack_6_block0_output"]
    use_weighted_sum = False
    return EfficientDet(**locals(), fpn_depth=6, head_depth=4, num_channels=200, anchor_scale=3, rescale_mode="tf", model_name="efficientdet_lite3x", **kwargs)


def EfficientDetLite4(input_shape=(640, 640, 3), freeze_backbone=False, num_classes=90, backbone=None, activation="relu6", pretrained="coco", **kwargs):
    if backbone is None:
        backbone = efficientnet.EfficientNetV1Lite4(input_shape=input_shape, num_classes=0, output_conv_filter=0, activation=activation)
        features_pick = ["stack_2_block3_output", "stack_4_block5_output", "stack_6_block0_output"]
    use_weighted_sum = False
    return EfficientDet(**locals(), fpn_depth=7, head_depth=4, num_channels=224, anchor_scale=3, rescale_mode="tf", model_name="efficientdet_lite4", **kwargs)
