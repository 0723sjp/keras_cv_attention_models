import os
import numpy as np
import tensorflow as tf
from keras_cv_attention_models.coco import data, anchors_func


def corners_to_center_yxhw_nd(ss):
    """input: [top, left, bottom, right], output: [center_h, center_w], [height, width]"""
    return (ss[:, :2] + ss[:, 2:]) * 0.5, ss[:, 2:] - ss[:, :2]


def yolor_assign_anchors(bboxes, labels, anchor_ratios, feature_sizes, anchor_aspect_thresh=4.0, overlap_offset=0.5):
    num_anchors, num_bboxes_true = anchor_ratios.shape[1], tf.shape(bboxes)[0]
    cum_feature_sizes = [0] + [ii[0] * ii[1] * num_anchors for ii in feature_sizes[:-1]]

    rr = []
    # for anchor_ratio, feature_size in zip(anchor_ratios, feature_sizes):
    for id in range(feature_sizes.shape[0]):
        # build_targets https://github.dev/WongKinYiu/yolor/blob/main/utils/loss.py#L127
        anchor_ratio, feature_size = anchor_ratios[id], feature_sizes[id]
        # pick by aspect ratio
        bboxes_centers, bboxes_hws = corners_to_center_yxhw_nd(bboxes)
        bboxes_centers, bboxes_hws = bboxes_centers * feature_size, bboxes_hws * feature_size
        aspect_ratio = tf.expand_dims(bboxes_hws, 0) / tf.expand_dims(anchor_ratio, 1)
        aspect_pick = tf.reduce_max(tf.maximum(aspect_ratio, 1 / aspect_ratio), axis=-1) < anchor_aspect_thresh
        anchors_pick = tf.repeat(tf.expand_dims(tf.range(num_anchors), -1), num_bboxes_true, axis=-1)[aspect_pick]
        aspect_picked_bboxes_labels = tf.concat([bboxes_centers, bboxes_hws, labels], axis=-1)
        aspect_picked_bboxes_labels = tf.repeat(tf.expand_dims(aspect_picked_bboxes_labels, 0), num_anchors, axis=0)[aspect_pick]

        # pick by centers
        centers = aspect_picked_bboxes_labels[:, :2]
        top, left = tf.unstack(tf.logical_and(centers % 1 < overlap_offset, centers > 1), axis=-1)
        bottom, right = tf.unstack(tf.logical_and(centers % 1 > (1 - overlap_offset), centers < (feature_size - 1)), axis=-1)
        anchors_pick_all = tf.concat([anchors_pick, anchors_pick[top], anchors_pick[left], anchors_pick[bottom], anchors_pick[right]], axis=0)
        matched_top, matched_left = aspect_picked_bboxes_labels[top], aspect_picked_bboxes_labels[left]
        matched_bottom, matched_right = aspect_picked_bboxes_labels[bottom], aspect_picked_bboxes_labels[right]
        matched_bboxes_all = tf.concat([aspect_picked_bboxes_labels, matched_top, matched_left, matched_bottom, matched_right], axis=0)

        # Use matched bboxes as indexes
        matched_bboxes_idx = tf.cast(aspect_picked_bboxes_labels[:, :2], "int32")
        matched_top_idx = tf.cast(matched_top[:, :2] - [overlap_offset, 0], "int32")
        matched_left_idx = tf.cast(matched_left[:, :2] - [0, overlap_offset], "int32")
        matched_bottom_idx = tf.cast(matched_bottom[:, :2] + [overlap_offset, 0], "int32")
        matched_right_idx = tf.cast(matched_right[:, :2] + [0, overlap_offset], "int32")
        matched_bboxes_idx_all = tf.concat([matched_bboxes_idx, matched_top_idx, matched_left_idx, matched_bottom_idx, matched_right_idx], axis=0)

        # Results
        centers_true = matched_bboxes_all[:, :2] - tf.cast(matched_bboxes_idx_all, matched_bboxes_all.dtype)
        bbox_labels_true = tf.concat([centers_true, matched_bboxes_all[:, 2:]], axis=-1)
        matched_bboxes_idx_all = tf.clip_by_value(matched_bboxes_idx_all, 0, tf.cast(feature_size, matched_bboxes_idx_all.dtype) - 1)
        # index = tf.concat([tf.zeros([anchors_pick_all.shape[0], 1], dtype='int32') + id, matched_bboxes_idx_all, tf.expand_dims(anchors_pick_all, 1)], axis=-1)

        index = matched_bboxes_idx_all[:, 0] * feature_size[1] * num_anchors + matched_bboxes_idx_all[:, 1] * num_anchors + anchors_pick_all
        index += cum_feature_sizes[id]

        rr.append(tf.concat([tf.cast(tf.expand_dims(index, 1), bbox_labels_true.dtype), bbox_labels_true], axis=-1))
    return tf.concat(rr, axis=0)  # [pyramid_feature_id, hieght, width, num_anchors]


def parse_targets_to_bboxes_labels(targets, batch_size, anchor_ratios, feature_sizes):
    bboxes_labels_with_batch_id = []
    image_indexes = targets[:, 0]
    for id in range(batch_size):
        cur_targets = targets[image_indexes == id]  # [id, label, center_left, center_top, width, height]
        # print(cur_targets.shape)
        cur_labels, cur_bboxes = cur_targets[:, 1:2], cur_targets[:, 2:]
        center_top, center_left, half_height, half_width = cur_bboxes[:, 1], cur_bboxes[:, 0], cur_bboxes[:, 3] / 2, cur_bboxes[:, 2] / 2
        cur_bboxes = np.stack([center_top - half_height, center_left - half_width, center_top + half_height, center_left + half_width], axis=-1)

        bboxes_labels = yolor_assign_anchors(cur_bboxes, cur_labels, anchor_ratios, feature_sizes, anchor_aspect_thresh=4.0, overlap_offset=0.5).numpy()
        cur_bboxes_labels_with_batch_id = np.concatenate([np.zeros([bboxes_labels.shape[0], 1]) + id, bboxes_labels], axis=-1)
        bboxes_labels_with_batch_id.append(cur_bboxes_labels_with_batch_id)

    return np.concatenate(bboxes_labels_with_batch_id, axis=0)


def dataset_gen(dataloader, anchor_ratios, feature_sizes, num_classes=80):
    for raw_imgs, raw_targets, paths, _ in dataloader:
        imgs = raw_imgs.numpy().transpose([0, 2, 3, 1]).astype("float32")
        bboxes_labels_with_batch_id = parse_targets_to_bboxes_labels(raw_targets.numpy(), imgs.shape[0], anchor_ratios, feature_sizes)

        # y_true = data.to_one_hot_with_class_mark(bboxes_labels_with_batch_id, num_classes)
        dest_boxes, anchor_classes = tf.split(bboxes_labels_with_batch_id, [-1, 1], axis=-1)
        one_hot_labels = tf.one_hot(tf.cast(anchor_classes[..., 0], "int32"), num_classes)  # [1, 81] -> [0, 80]
        one_hot_labels = tf.cast(one_hot_labels, dest_boxes.dtype)
        y_true = tf.concat([dest_boxes, one_hot_labels], axis=-1)
        yield imgs, y_true


def init_dataset(
    data_name="../coco",  # dataset path
    input_shape=(256, 256),
    batch_size=64,
    info_only=False,
    anchors_mode="yolor",
    anchor_pyramid_levels=[3, 5],
    rescale_mode="raw01",  # rescale mode, ["tf", "torch"], or specific `(mean, std)` like `(128.0, 128.0)`
    hyp=None,
    **kwargs,  # not using, just recieving
):
    from collections import namedtuple
    from utils.datasets import create_dataloader

    train_path = os.path.join(data_name, "train2017.txt")
    if info_only:
        with open(train_path) as ff:
            total_images = len(ff.readlines())
        steps_per_epoch = int(tf.math.ceil(total_images / float(batch_size)))
        num_classes = 80
        return total_images, num_classes, steps_per_epoch

    default_hyp = {
        "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
        "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
        "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
        "degrees": 0.0,  # image rotation (+/- deg)
        "translate": 0.1,  # image translation (+/- fraction), 0.1 for hyp.scratch.640.yaml, 0.5 for hyp.scratch.1280.yaml
        "scale": 0.9,  # image scale (+/- gain), 0.9 for hyp.scratch.640.yaml, 0.5 for hyp.scratch.1280.yaml
        "shear": 0.0,  # image shear (+/- deg)
        "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
        "flipud": 0.0,  # image flip up-down (probability)
        "fliplr": 0.5,  # image flip left-right (probability)
        "mosaic": 1.0,  # image mosaic (probability)
        "mixup": 0.0,  # image mixup (probability)
    }
    hyp = default_hyp if hyp is None else hyp

    opt = namedtuple('opt', ['single_cls'])(False)
    grid_size = 64  # grid size (max stride)
    train_loader, dataset = create_dataloader(path=train_path, imgsz=input_shape[0], batch_size=batch_size, stride=grid_size, opt=opt, hyp=hyp, augment=True)
    num_classes = np.concatenate(dataset.labels, 0)[:, 0].max().astype("int") + 1
    total_images, steps_per_epoch = len(dataset), len(train_loader)

    anchor_ratios, feature_sizes = anchors_func.get_yolor_anchors(input_shape[:2], anchor_pyramid_levels, is_for_training=True)
    output_signature = (
        tf.TensorSpec(shape=(None, *input_shape[:2], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 9 + num_classes), dtype=tf.float32)
    )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_gen(train_loader, anchor_ratios, feature_sizes, num_classes), output_signature=output_signature
    )

    AUTOTUNE = tf.data.AUTOTUNE
    mean, std = data.init_mean_std_by_rescale_mode(rescale_mode)
    rescaling = lambda xx: (xx - mean) / std
    train_dataset = train_dataset.map(lambda xx, yy: (rescaling(xx), yy), num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.apply(tf.data.experimental.assert_cardinality(steps_per_epoch)).with_options(options)

    test_dataset = None
    return train_dataset, test_dataset, total_images, num_classes, steps_per_epoch
