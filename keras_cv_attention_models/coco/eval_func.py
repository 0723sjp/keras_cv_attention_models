import tensorflow as tf
import tensorflow_datasets as tfds
from keras_cv_attention_models.coco import data
from tqdm import tqdm


class DecodePredictions:
    def __init__(self, input_shape=(512, 512, 3), pyramid_levels=[3, 7], anchor_scale=4, **kwargs):
        self.anchor_scale, self.kwargs = anchor_scale, kwargs
        self.pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
        if input_shape[0] is not None:
            self.__init_anchor__(input_shape)
        else:
            self.anchors = None

    def __init_anchor__(self, input_shape):
        self.anchors = data.get_anchors(input_shape=input_shape, pyramid_levels=self.pyramid_levels, anchor_scale=self.anchor_scale, **self.kwargs)

    def __topk_class_boxes_single__(self, pred, topk=5000):
        # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L82
        bbox_outputs, class_outputs = pred[:, :4], pred[:, 4:]
        num_classes = class_outputs.shape[-1]
        class_outputs_flatten = tf.reshape(class_outputs, -1)
        _, class_topk_indices = tf.nn.top_k(class_outputs_flatten, k=topk, sorted=False)
        # get original indices for class_outputs, original_indices_hh -> picking indices, original_indices_ww -> picked labels
        original_indices_hh, original_indices_ww = class_topk_indices // num_classes, class_topk_indices % num_classes
        class_indices = tf.stack([original_indices_hh, original_indices_ww], axis=-1)
        scores_topk = tf.gather_nd(class_outputs, class_indices)
        bboxes_topk = tf.gather(bbox_outputs, original_indices_hh)
        return bboxes_topk, scores_topk, original_indices_ww, original_indices_hh

    def __nms_per_class__(self, bbs, ccs, labels, score_threshold=0.3, iou_threshold=0.5, soft_nms_sigma=0.5, max_output_size=100):
        # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L409
        rrs = []
        for ii in tf.unique(labels)[0]:
            pick = tf.where(labels == ii)[:, 0]
            bb, cc = tf.gather(bbs, pick), tf.gather(ccs, pick)
            rr, nms_scores = tf.image.non_max_suppression_with_scores(bb, cc, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
            bb_nms = tf.gather(bb, rr)
            rrs.append(tf.concat([bb_nms, tf.ones([bb_nms.shape[0], 1]) * tf.cast(ii, bb_nms.dtype), tf.expand_dims(nms_scores, 1)], axis=-1))
        rrs = tf.concat(rrs, axis=0)
        if tf.shape(rrs)[0] > max_output_size:
            score_top_k = tf.argsort(rrs[:, -1], direction="DESCENDING")[:max_output_size]
            rrs = tf.gather(rrs, score_top_k)
        bboxes, labels, scores = rrs[:, :4], rrs[:, 4], rrs[:, -1]
        return bboxes, labels, scores

    def __nms_global__(self, bbs, ccs, labels, score_threshold=0.3, iou_threshold=0.5, soft_nms_sigma=0.5, max_output_size=100):
        rr, nms_scores = tf.image.non_max_suppression_with_scores(bbs, ccs, max_output_size, iou_threshold, score_threshold, soft_nms_sigma)
        return tf.gather(bbs, rr), tf.gather(labels, rr), nms_scores

    def __decode_single__(self, pred, score_threshold=0.3, iou_or_sigma=0.5, max_output_size=100, method="gaussian", mode="global", topk=-1, input_shape=None):
        # https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L159
        if input_shape is not None:
            self.__init_anchor__(input_shape)

        if topk > 0:
            bbs, ccs, labels, picking_indices = self.__topk_class_boxes_single__(pred, topk)
            anchors = tf.gather(self.anchors, picking_indices)
        else:
            bbs, ccs, labels = pred[:, :4], tf.reduce_max(pred[:, 4:], axis=-1), tf.argmax(pred[:, 4:], axis=-1)
            anchors = self.anchors

        bbs_decoded = data.decode_bboxes(bbs, anchors)
        iou_threshold, soft_nms_sigma = (1.0, iou_or_sigma / 2) if method.lower() == "gaussian" else (iou_or_sigma, 0.0)

        if mode == "per_class":
            return self.__nms_per_class__(bbs_decoded, ccs, labels, score_threshold, iou_threshold, soft_nms_sigma, max_output_size)
        else:
            return self.__nms_global__(bbs_decoded, ccs, labels, score_threshold, iou_threshold, soft_nms_sigma, max_output_size)

    def __call__(self, preds, score_threshold=0.3, iou_or_sigma=0.5, max_output_size=100, method="gaussian", mode="global", topk=-1, input_shape=None):
        """
        https://github.com/google/automl/tree/master/efficientdet/tf2/postprocess.py#L159
        iou_or_sigma: means `soft_nms_sigma` if method is "gaussian", else `iou_threshold`.
        method: "gaussian" or "hard".
        mode: "global" or "per_class".
        topk: `> 0` value for picking topk preds using scores.
        """
        preds = preds if len(preds.shape) == 3 else [preds]
        return [self.__decode_single__(pred, score_threshold, iou_or_sigma, max_output_size, method, mode, topk, input_shape) for pred in preds]


def scale_bboxes_back_single(bboxes, image_shape, scale, target_shape):
    height, width = target_shape[0] / scale, target_shape[1] / scale
    bboxes *= [height, width, height, width]
    bboxes = tf.clip_by_value(bboxes, 0, clip_value_max=[image_shape[0], image_shape[1], image_shape[0], image_shape[1]])
    # [top, left, bottom, right] -> [left, top, width, height]
    bboxes = tf.stack([bboxes[:, 1], bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0]], axis=-1)
    return bboxes


def init_eval_dataset(data_name="coco/2017", target_shape=(512, 512), batch_size=8, rescale_mode="torch", resize_method="bilinear", resize_antialias=False):
    mean, std = data.init_mean_std_by_rescale_mode(rescale_mode)
    rescaling = lambda xx: (xx - mean) / std

    ds = tfds.load(data_name, with_info=False)["validation"]
    resize_func = lambda image: data.aspect_aware_resize_and_crop_image(image, target_shape, method=resize_method, antialias=resize_antialias)
    # ds: [resized_image, original_image_shape, scale, image_id]
    ds = ds.map(lambda datapoint: (*resize_func(rescaling(tf.cast(datapoint["image"], tf.float32))), tf.shape(datapoint["image"])[:2], datapoint["image/id"]))
    ds = ds.batch(batch_size)
    return ds


def model_eval_results(model, eval_dataset, pred_decoder, score_threshold=0.001, method="gaussian", mode="per_class", topk=5000):
    target_shape = (eval_dataset.element_spec[0].shape[1], eval_dataset.element_spec[0].shape[2])
    num_classes = model.output_shape[-1] - 4
    to_90_labels = (lambda label: label + 1) if num_classes == 90 else (lambda label: data.COCO_80_to_90_LABEL_DICT[label] + 1)
    # Format: [image_id, x, y, width, height, score, class]
    to_coco_eval_single = lambda image_id, bbox, label, score: [image_id.numpy(), *bbox.numpy().tolist(), score.numpy(), to_90_labels(label.numpy())]

    results = []
    for images, scales, original_image_shapes, image_ids in tqdm(eval_dataset):
        preds = model(images)
        # decoded_preds: [[bboxes, labels, scores], [bboxes, labels, scores], ...]
        decoded_preds = pred_decoder(preds, score_threshold=score_threshold, method=method, mode=mode, topk=topk)

        for rr, image_shape, scale, image_id in zip(decoded_preds, original_image_shapes, scales, image_ids):  # Loop on batch
            bboxes, labels, scores = rr
            bboxes = scale_bboxes_back_single(bboxes, image_shape, scale, target_shape)
            results.extend([to_coco_eval_single(image_id, bb, cc, ss) for bb, cc, ss in zip(bboxes, labels, scores)])  # Loop on prediction results
    return tf.convert_to_tensor(results).numpy()


def coco_evaluation(detection_results, annotation_file=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if annotation_file is None:
        url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/coco_annotations_instances_val2017.json"
        annotation_file = tf.keras.utils.get_file(origin=url)
    coco_gt = COCO(annotation_file)
    image_ids = list(set(detection_results[:, 0]))
    coco_dt = coco_gt.loadRes(detection_results)
    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def run_coco_evaluation(
    model,
    data_name="coco/2017",  # init_eval_dataset
    input_shape=None,
    batch_size=8,
    rescale_mode="torch",
    resize_method="bilinear",
    resize_antialias=False,
    score_threshold=0.001,  # model_eval_results
    method="gaussian",
    mode="per_class",
    topk=5000,
    annotation_file=None,  # coco_evaluation
    pyramid_levels=[3, 7],  # get_anchors
    anchor_scale=4,
    **anchor_kwargs,
):
    input_shape = model.input_shape[1:-1] if input_shape is None else input_shape
    eval_dataset = eval_func.init_eval_dataset(data_name, input_shape, batch_size, rescale_mode, resize_method, resize_antialias)
    pred_decoder = model.decode_predictions if hasattr(model, "decode_predictions") else eval_func.DecodePredictions(input_shape)
    detection_results = eval_func.model_eval_results(model, eval_dataset, pred_decoder, score_threshold, method, mode, topk)
    return eval_func.coco_evaluation(detection_results, annotation_file)


if __name__ == "__test__":
    from keras_cv_attention_models.coco import eval_func
    from keras_cv_attention_models import efficientdet

    model = efficientdet.EfficientDetD0(pretrained="efficientdet_d0.h5")
    input_shape = model.input_shape[1:-1]
    ds = eval_func.init_eval_dataset(target_shape=input_shape)
    pred_decoder = model.decode_predictions if hasattr(model, "decode_predictions") else eval_func.DecodePredictions(input_shape)
    detection_results = eval_func.model_eval_results(model, ds, pred_decoder)
    eval_func.coco_evaluation(detection_results)
