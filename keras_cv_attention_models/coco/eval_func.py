import os
import tensorflow as tf
from keras_cv_attention_models.coco import anchors_func, data
from tqdm import tqdm


@tf.keras.utils.register_keras_serializable(package="kecam/coco")
class DecodePredictions(tf.keras.layers.Layer):
    """
    The most simple version decoding prediction and NMS:

    >>> from keras_cv_attention_models import efficientdet, test_images
    >>> model = efficientdet.EfficientDetD0()
    >>> preds = model(model.preprocess_input(test_images.dog()))

    # Decode and NMS
    >>> from keras_cv_attention_models import coco
    >>> input_shape = model.input_shape[1:-1]
    >>> anchors = coco.get_anchors(input_shape=input_shape, pyramid_levels=[3, 7], anchor_scale=4)
    >>> dd = coco.decode_bboxes(preds[0], anchors).numpy()
    >>> rr = tf.image.non_max_suppression(dd[:, :4], dd[:, 4:].max(-1), score_threshold=0.3, max_output_size=15, iou_threshold=0.5)
    >>> dd_nms = tf.gather(dd, rr).numpy()
    >>> bboxes, labels, scores = dd_nms[:, :4], dd_nms[:, 4:].argmax(-1), dd_nms[:, 4:].max(-1)
    >>> print(f"{bboxes = }, {labels = }, {scores = }")
    >>> # bboxes = array([[0.433231  , 0.54432285, 0.8778939 , 0.8187578 ]], dtype=float32), labels = array([17]), scores = array([0.85373735], dtype=float32)
    """

    def __init__(
        self,
        input_shape=512,
        pyramid_levels=[3, 7],
        anchors_mode=None,
        use_object_scores="auto",
        anchor_scale="auto",
        aspect_ratios=(1, 2, 0.5),
        num_scales=3,
        score_threshold=0.3,  # decode parameter, can be set new value in `self.call`
        iou_or_sigma=0.5,  # decode parameter, can be set new value in `self.call`
        max_output_size=100,  # decode parameter, can be set new value in `self.call`
        method="hard",  # decode parameter, can be set new value in `self.call`
        mode="global",  # decode parameter, can be set new value in `self.call`
        topk=0,  # decode parameter, can be set new value in `self.call`
        use_static_output=False,  # Set to True if using this as an actual layer, especially for converting tflite
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
        use_object_scores, num_anchors, anchor_scale = anchors_func.get_anchors_mode_parameters(anchors_mode, use_object_scores, "auto", anchor_scale)
        self.aspect_ratios, self.num_scales = aspect_ratios, num_scales
        self.anchors_mode, self.use_object_scores, self.anchor_scale = anchors_mode, use_object_scores, anchor_scale  # num_anchors not using
        if input_shape is not None and (isinstance(input_shape, (list, tuple)) and input_shape[0] is not None):
            self.__init_anchor__(input_shape)
        else:
            self.anchors = None
        self.__input_shape__ = input_shape
        self.use_static_output = use_static_output
        self.nms_kwargs = {
            "score_threshold": score_threshold,
            "iou_or_sigma": iou_or_sigma,
            "max_output_size": max_output_size,
            "method": method,
            "mode": mode,
            "topk": topk,
        }
        super().build(input_shape)

    def __init_anchor__(self, input_shape):
        input_shape = input_shape[:2] if isinstance(input_shape, (list, tuple)) else (input_shape, input_shape)
        if self.anchors_mode == anchors_func.ANCHOR_FREE_MODE:
            self.anchors = anchors_func.get_anchor_free_anchors(input_shape, self.pyramid_levels)
        elif self.anchors_mode == anchors_func.YOLOR_MODE:
            self.anchors = anchors_func.get_yolor_anchors(input_shape, self.pyramid_levels)
        else:
            grid_zero_start = False
            self.anchors = anchors_func.get_anchors(input_shape, self.pyramid_levels, self.aspect_ratios, self.num_scales, self.anchor_scale, grid_zero_start)
        self.__input_shape__ = input_shape
        return self.anchors

    def __topk_class_boxes_single__(self, pred, topk=5000):
        # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L82
        bbox_outputs, class_outputs = pred[:, :4], pred[:, 4:]
        num_classes = class_outputs.shape[-1]
        class_outputs_flatten = tf.reshape(class_outputs, -1)
        topk = class_outputs_flatten.shape[0] if topk == -1 else topk  # select all if -1
        _, class_topk_indices = tf.nn.top_k(class_outputs_flatten, k=topk, sorted=False)
        # get original indices for class_outputs, original_indices_hh -> picking indices, original_indices_ww -> picked labels
        original_indices_hh, original_indices_ww = class_topk_indices // num_classes, class_topk_indices % num_classes
        class_indices = tf.stack([original_indices_hh, original_indices_ww], axis=-1)
        scores_topk = tf.gather_nd(class_outputs, class_indices)
        bboxes_topk = tf.gather(bbox_outputs, original_indices_hh)
        return bboxes_topk, scores_topk, original_indices_ww, original_indices_hh

    # def __nms_per_class__(self, bbs, ccs, labels, score_threshold=0.3, iou_threshold=0.5, soft_nms_sigma=0.5, max_output_size=100):
    #     # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L409
    #     # Not using, same result with `torchvision.ops.batched_nms`
    #     rrs = []
    #     for ii in tf.unique(labels)[0]:
    #         pick = tf.where(labels == ii)
    #         bb, cc = tf.gather_nd(bbs, pick), tf.gather_nd(ccs, pick)
    #         rr, nms_scores = tf.image.non_max_suppression_with_scores(bb, cc, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
    #         bb_nms = tf.gather(bb, rr)
    #         rrs.append(tf.concat([bb_nms, tf.ones([bb_nms.shape[0], 1]) * tf.cast(ii, bb_nms.dtype), tf.expand_dims(nms_scores, 1)], axis=-1))
    #     rrs = tf.concat(rrs, axis=0)
    #     if tf.shape(rrs)[0] > max_output_size:
    #         score_top_k = tf.argsort(rrs[:, -1], direction="DESCENDING")[:max_output_size]
    #         rrs = tf.gather(rrs, score_top_k)
    #     bboxes, labels, scores = rrs[:, :4], rrs[:, 4], rrs[:, -1]
    #     return bboxes.numpy(), labels.numpy(), scores.numpy()

    def __nms_per_class__(self, bbs, ccs, labels, score_threshold=0.3, iou_threshold=0.5, soft_nms_sigma=0.5, max_output_size=100):
        # From torchvision.ops.batched_nms strategy: in order to perform NMS independently per class. we add an offset to all the boxes.
        # The offset is dependent only on the class idx, and is large enough so that boxes from different classes do not overlap
        # Same result with per_class method: https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L409
        cls_offset = tf.cast(labels, bbs.dtype) * (tf.reduce_max(bbs) + 1)
        bbs_per_class = bbs + tf.expand_dims(cls_offset, -1)
        rr, nms_scores = tf.image.non_max_suppression_with_scores(bbs_per_class, ccs, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
        return tf.gather(bbs, rr), tf.gather(labels, rr), nms_scores

    def __nms_global__(self, bbs, ccs, labels, score_threshold=0.3, iou_threshold=0.5, soft_nms_sigma=0.5, max_output_size=100):
        rr, nms_scores = tf.image.non_max_suppression_with_scores(bbs, ccs, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
        return tf.gather(bbs, rr), tf.gather(labels, rr), nms_scores

    def __object_score_split__(self, pred):
        return pred[:, :-1], pred[:, -1]  # May overwrite

    def __to_static__(self, bboxs, lables, confidences, max_output_size=100):
        indices = tf.expand_dims(tf.range(tf.shape(bboxs)[0]), -1)
        lables = tf.cast(lables, bboxs.dtype)
        concated = tf.concat([bboxs, tf.expand_dims(lables, -1), tf.expand_dims(confidences, -1)], axis=-1)
        concated = tf.tensor_scatter_nd_update(tf.zeros([max_output_size, concated.shape[-1]], dtype=bboxs.dtype), indices, concated)
        return concated

    def __decode_single__(self, pred, score_threshold=0.3, iou_or_sigma=0.5, max_output_size=100, method="hard", mode="global", topk=0, input_shape=None):
        # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L159
        pred = tf.cast(pred, tf.float32)
        if input_shape is not None:
            self.__init_anchor__(input_shape)

        if self.use_object_scores:  # YOLO outputs: [bboxes, classses_score, object_score]
            pred, object_scores = self.__object_score_split__(pred)

        if topk != 0:
            bbs, ccs, labels, picking_indices = self.__topk_class_boxes_single__(pred, topk)
            anchors = tf.gather(self.anchors, picking_indices)
            if self.use_object_scores:
                ccs *= tf.gather(object_scores, picking_indices)
        else:
            bbs, ccs, labels = pred[:, :4], tf.reduce_max(pred[:, 4:], axis=-1), tf.argmax(pred[:, 4:], axis=-1)
            anchors = self.anchors
            if self.use_object_scores:
                ccs *= object_scores

        bbs_decoded = anchors_func.decode_bboxes(bbs, anchors)
        iou_threshold, soft_nms_sigma = (1.0, iou_or_sigma / 2) if method.lower() == "gaussian" else (iou_or_sigma, 0.0)

        if mode == "per_class":
            bboxs, lables, confidences = self.__nms_per_class__(bbs_decoded, ccs, labels, score_threshold, iou_threshold, soft_nms_sigma, max_output_size)
        else:
            bboxs, lables, confidences = self.__nms_global__(bbs_decoded, ccs, labels, score_threshold, iou_threshold, soft_nms_sigma, max_output_size)

        return self.__to_static__(bboxs, lables, confidences, max_output_size) if self.use_static_output else (bboxs, lables, confidences)

    def call(self, preds, input_shape=None, training=False, **nms_kwargs):
        """
        https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L159

        input_shape: actual input shape if model using dynamic input shape `[None, None, 3]`.
        nms_kwargs:
          score_threshold: float value in (0, 1), min score threshold, lower output score will be excluded. Default 0.3.
          iou_or_sigma: means `soft_nms_sigma` if method is "gaussian", else `iou_threshold`. Default 0.5.
          max_output_size: max output size for `tf.image.non_max_suppression_with_scores`. Default 100.
              If use_static_output=True, fixed output shape will be `[batch, max_output_size, 6]`.
          method: "gaussian" or "hard".  Default "hard".
          mode: "global" or "per_class". "per_class" is strategy from `torchvision.ops.batched_nms`. Default "global".
          topk: Using topk highest scores, each bbox may have multi labels. Set `0` to disable, `-1` using all. Default 0.
        """
        self.nms_kwargs.update(nms_kwargs)
        if self.use_static_output:
            return tf.map_fn(lambda xx: self.__decode_single__(xx, **nms_kwargs), preds)
        elif len(preds.shape) == 3:
            return [self.__decode_single__(pred, **self.nms_kwargs, input_shape=input_shape) for pred in preds]
        else:
            return self.__decode_single__(preds, **self.nms_kwargs, input_shape=input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.__input_shape__,
                "pyramid_levels": self.pyramid_levels,
                "anchors_mode": self.anchors_mode,
                "use_object_scores": self.use_object_scores,
                "anchor_scale": self.anchor_scale,
                "aspect_ratios": self.aspect_ratios,
                "num_scales": self.num_scales,
                "use_static_output": self.use_static_output,
            }
        )
        config.update(self.nms_kwargs)
        return config


def scale_bboxes_back_single(bboxes, image_shape, scale, pad_top, pad_left, target_shape):
    # height, width = target_shape[0] / scale, target_shape[1] / scale
    # bboxes *= [height, width, height, width]
    bboxes *= [target_shape[0], target_shape[1], target_shape[0], target_shape[1]]
    bboxes -= [pad_top, pad_left, pad_top, pad_left]
    bboxes /= scale
    bboxes = tf.clip_by_value(bboxes, 0, clip_value_max=[image_shape[0], image_shape[1], image_shape[0], image_shape[1]])
    # [top, left, bottom, right] -> [left, top, width, height]
    bboxes = tf.stack([bboxes[:, 1], bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0]], axis=-1)
    return bboxes


def image_process(image, target_shape, mean, std, resize_method="bilinear", resize_antialias=False, use_bgr_input=False, letterbox_pad=-1):
    if len(image.shape) < 2:
        image = data.tf_imread(image)  # it's image path
    original_image_shape = tf.shape(image)[:2]
    image = tf.cast(image, "float32")
    image = (image - mean) / std  # automl behavior: rescale -> resize
    image, scale, pad_top, pad_left = data.aspect_aware_resize_and_crop_image(
        image, target_shape, letterbox_pad=letterbox_pad, method=resize_method, antialias=resize_antialias
    )
    if use_bgr_input:
        image = image[:, :, ::-1]
    return image, scale, pad_top, pad_left, original_image_shape


def init_eval_dataset(
    data_name="coco/2017",
    input_shape=(512, 512),
    batch_size=8,
    rescale_mode="torch",
    resize_method="bilinear",
    resize_antialias=False,
    letterbox_pad=-1,
    use_bgr_input=False,
):
    import tensorflow_datasets as tfds

    dataset = data.detection_dataset_from_custom_json(data_name) if data_name.endswith(".json") else tfds.load(data_name)
    ds = dataset.get("validation", dataset.get("test", None))

    mean, std = data.init_mean_std_by_rescale_mode(rescale_mode)
    __image_process__ = lambda image: image_process(image, input_shape, mean, std, resize_method, resize_antialias, use_bgr_input, letterbox_pad)
    # ds: [resized_image, scale, pad_top, pad_left, original_image_shape, image_id]
    ds = ds.map(lambda datapoint: (*__image_process__(datapoint["image"]), datapoint.get("image/id", datapoint["image"])))
    ds = ds.batch(batch_size)
    return ds


def model_detection_and_decode(model, eval_dataset, pred_decoder, nms_kwargs={}, is_coco=True, image_id_map=None):
    target_shape = (eval_dataset.element_spec[0].shape[1], eval_dataset.element_spec[0].shape[2])
    num_classes = model.output_shape[-1] - 4
    if is_coco:
        to_91_labels = (lambda label: label + 1) if num_classes >= 90 else (lambda label: data.COCO_80_to_90_LABEL_DICT[label] + 1)
    else:
        to_91_labels = lambda label: label
    # Format: [image_id, x, y, width, height, score, class]
    to_coco_eval_single = lambda image_id, bbox, label, score: [image_id, *bbox.tolist(), score, to_91_labels(label)]

    results = []
    for images, scales, pad_tops, pad_lefts, original_image_shapes, image_ids in tqdm(eval_dataset):
        preds = model(images)
        preds = tf.cast(preds, tf.float32)
        # decoded_preds: [[bboxes, labels, scores], [bboxes, labels, scores], ...]
        decoded_preds = pred_decoder(preds, **nms_kwargs)

        # Loop on batch
        for rr, image_shape, scale, pad_top, pad_left, image_id in zip(decoded_preds, original_image_shapes, scales, pad_tops, pad_lefts, image_ids):
            bboxes, labels, scores = rr
            image_id, bboxes, labels, scores = image_id.numpy(), bboxes.numpy(), labels.numpy(), scores.numpy()
            if image_id_map is not None:
                image_id = image_id_map[image_id.decode() if isinstance(image_id, bytes) else image_id]
            bboxes = scale_bboxes_back_single(bboxes, image_shape, scale, pad_top, pad_left, target_shape).numpy()
            results.extend([to_coco_eval_single(image_id, bb, cc, ss) for bb, cc, ss in zip(bboxes, labels, scores)])  # Loop on prediction results
    return tf.convert_to_tensor(results).numpy()


class COCOEvaluation:
    def __init__(self, annotations=None):
        from pycocotools.coco import COCO

        if annotations is None:
            url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/coco_annotations_instances_val2017.json"
            annotations = tf.keras.utils.get_file(origin=url)

        if isinstance(annotations, dict):  # json already loaded as dict
            coco_gt = COCO()
            coco_gt.dataset = annotations
            coco_gt.createIndex()
        else:
            coco_gt = COCO(annotations)
        self.coco_gt = coco_gt

    def __call__(self, detection_results):
        from pycocotools.cocoeval import COCOeval

        image_ids = [ii["image_id"] for ii in detection_results] if isinstance(detection_results[0], dict) else [ii[0] for ii in detection_results]
        image_ids = list(set(image_ids))
        print("len(image_ids) =", len(image_ids))
        coco_dt = self.coco_gt.loadRes(detection_results)
        coco_eval = COCOeval(cocoGt=self.coco_gt, cocoDt=coco_dt, iouType="bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval


def to_coco_json(detection_results, save_path, indent=2):
    import json

    __to_coco_json__ = lambda xx: {"image_id": int(xx[0]), "bbox": [float(ii) for ii in xx[1:5]], "score": float(xx[5]), "category_id": int(xx[6])}
    aa = [__to_coco_json__(ii) for ii in detection_results]
    with open(save_path, "w") as ff:
        json.dump(aa, ff, indent=indent)


def to_coco_annotation(json_path):
    import json
    from PIL import Image

    with open(json_path, "r") as ff:
        aa = json.load(ff)

    # int conversion just in case key is str
    categories = {int(kk): vv for kk, vv in aa["indices_2_labels"].items()} if "indices_2_labels" in aa else {}
    annotations, images, image_id_map = [], [], {}
    for image_id, ii in enumerate(aa.get("validation", aa.get("test", []))):
        width, height = Image.open(ii["image"]).size  # For decoding bboxes, not actually openning images
        for bb, label in zip(ii["objects"]["bbox"], ii["objects"]["label"]):
            # bb [top, left, bottom, right] in [0, 1] -> [left, top, bbox_width, bbox_height] with actual coordinates
            top = bb[0] * height
            left = bb[1] * width
            bbox_height = bb[2] * height - top
            bbox_width = bb[3] * width - left
            bb = [left, top, bbox_width, bbox_height]
            area = bbox_width * bbox_height  # Actual area in COCO is the segmentation area, doesn't matter in detection mission

            label = int(label)
            annotations.append({"bbox": bb, "category_id": label, "image_id": image_id, "id": len(annotations), "iscrowd": 0, "area": area})
            if label not in categories:
                categories[label] = str(len(categories))
        images.append({"id": image_id, "file_name": ii["image"], "height": height, "width": width})
        image_id_map[ii["image"]] = image_id
    categories = [{"id": kk, "name": vv} for kk, vv in categories.items()]
    return {"images": images, "annotations": annotations, "categories": categories}, image_id_map


# Wrapper a callback for using in training
class COCOEvalCallback(tf.keras.callbacks.Callback):
    """
    Basic test:
    >>> from keras_cv_attention_models import efficientdet, coco
    >>> model = efficientdet.EfficientDetD0()
    >>> ee = coco.eval_func.COCOEvalCallback(batch_size=4, model_basic_save_name='test', rescale_mode='raw', anchors_mode="anchor_free")
    >>> ee.model = model
    >>> ee.on_epoch_end()
    """

    def __init__(
        self,
        data_name="coco/2017",  # [init_eval_dataset parameters]
        batch_size=8,
        resize_method="bilinear",
        resize_antialias=False,
        rescale_mode="auto",
        letterbox_pad=-1,
        use_bgr_input=False,
        take_samples=-1,
        nms_score_threshold=0.001,  # [model_detection_and_decode parameters]
        nms_iou_or_sigma=0.5,
        nms_max_output_size=100,
        nms_method="gaussian",
        nms_mode="per_class",
        nms_topk=5000,
        anchors_mode="auto",  # [model anchors related parameters]
        anchor_scale=4,  # Init anchors for model prediction. "auto" means 1 if (anchors_mode=="anchor_free" or anchors_mode=="yolor"), else 4
        aspect_ratios=(1, 2, 0.5),  # For efficientdet anchors only
        num_scales=3,  # For efficientdet anchors only
        annotation_file=None,
        save_json=None,
        start_epoch=0,  # [trainign callbacks parameters]
        frequency=1,
        model_basic_save_name=None,
    ):
        super().__init__()
        self.anchors_mode = anchors_mode
        self.take_samples, self.annotation_file, self.start_epoch, self.frequency = take_samples, annotation_file, start_epoch, frequency
        self.save_json, self.model_basic_save_name, self.save_path, self.item_key = save_json, model_basic_save_name, "checkpoints", "val_ap_ar"
        self.data_name = data_name

        self.dataset_kwargs = {
            "data_name": data_name,
            "batch_size": batch_size,
            "rescale_mode": rescale_mode,
            "resize_method": resize_method,
            "resize_antialias": resize_antialias,
            "letterbox_pad": letterbox_pad,
            "use_bgr_input": use_bgr_input,
        }
        self.nms_kwargs = {
            "score_threshold": nms_score_threshold,
            "iou_or_sigma": nms_iou_or_sigma,
            "max_output_size": nms_max_output_size,
            "method": nms_method,
            "mode": nms_mode,
            "topk": nms_topk,
        }
        self.anchor_kwargs = {
            "anchor_scale": anchor_scale,
            "aspect_ratios": aspect_ratios,
            "num_scales": num_scales,
        }
        self.efficient_det_num_anchors = len(aspect_ratios) * num_scales

        self.is_coco = True if data_name.startswith("coco") and not data_name.endswith(".json") else False
        if self.data_name.endswith(".json") and self.annotation_file is None:
            self.annotation_file, self.image_id_map = to_coco_annotation(self.data_name)
        else:
            self.image_id_map = None

        self.built = False

    def build(self, input_shape, output_shape):
        input_shape = (int(input_shape[1]), int(input_shape[2]))
        self.eval_dataset = init_eval_dataset(input_shape=input_shape, **self.dataset_kwargs)
        if self.anchors_mode is None or self.anchors_mode == "auto":
            self.anchors_mode, num_anchors = anchors_func.get_anchors_mode_by_anchors(input_shape, total_anchors=output_shape[1])
        elif self.anchors_mode == anchors_func.EFFICIENTDET_MODE:
            num_anchors = self.efficient_det_num_anchors
        else:
            num_anchors = anchors_func.NUM_ANCHORS.get(self.anchors_mode, 9)
        pyramid_levels = anchors_func.get_pyramid_levels_by_anchors(input_shape, total_anchors=output_shape[1], num_anchors=num_anchors)
        print("\n>>>> [COCOEvalCallback] input_shape: {}, pyramid_levels: {}, anchors_mode: {}".format(input_shape, pyramid_levels, self.anchors_mode))
        # print(">>>>", self.dataset_kwargs)
        # print(">>>>", self.nms_kwargs)

        self.pred_decoder = DecodePredictions(input_shape, pyramid_levels, self.anchors_mode, **self.anchor_kwargs)

        # Training saving best
        if self.model_basic_save_name is not None:
            self.monitor_save = os.path.join(self.save_path, self.model_basic_save_name + "_epoch_{}_" + self.item_key + "_{}.h5")
            self.monitor_save_re = self.monitor_save.format("*", "*")
            self.is_better = lambda cur, pre: cur >= pre
            self.pre_best = -1e5

        self.coco_evaluation = COCOEvaluation(self.annotation_file)
        self.built = True

    def on_epoch_end(self, epoch=0, logs=None):
        if not self.built:
            if self.dataset_kwargs["rescale_mode"] == "auto":
                self.dataset_kwargs["rescale_mode"] = getattr(self.model, "rescale_mode", "torch")
            self.build(self.model.input_shape, self.model.output_shape)

        if epoch + 1 < self.start_epoch or epoch % self.frequency != 0:
            return

        # pred_decoder = self.model.decode_predictions
        eval_dataset = self.eval_dataset.take(self.take_samples) if self.take_samples > 0 else self.eval_dataset
        detection_results = model_detection_and_decode(self.model, eval_dataset, self.pred_decoder, self.nms_kwargs, self.is_coco, self.image_id_map)
        try:
            coco_eval = self.coco_evaluation(detection_results)
        except:
            print(">>>> Error in running coco_evaluation")
            coco_eval = None
            data_name = self.data_name.replace("/", "_")
            self.save_json = "{}_{}_detection_results_error.json".format(self.model.name, data_name) if self.save_json is None else self.save_json

        if self.save_json is not None:
            to_coco_json(detection_results, self.save_json)
            print(">>>> Detection results saved to:", self.save_json)

        if hasattr(self.model, "history") and hasattr(self.model.history, "history"):
            self.model.history.history.setdefault(self.item_key, []).append(coco_eval.stats.tolist())

        # Training save best
        cur_ap = coco_eval.stats[0] if coco_eval is not None else 0
        if self.model_basic_save_name is not None and self.is_better(cur_ap, self.pre_best):
            self.pre_best = cur_ap
            pre_monitor_saves = tf.io.gfile.glob(self.monitor_save_re)
            # tf.print(">>>> pre_monitor_saves:", pre_monitor_saves)
            if len(pre_monitor_saves) != 0:
                os.remove(pre_monitor_saves[0])
            monitor_save = self.monitor_save.format(epoch + 1, "{:.4f}".format(cur_ap))
            tf.print("\n>>>> Save best to:", monitor_save)
            if self.model is not None:
                self.model.save(monitor_save)

        return coco_eval
