import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img


def random_crop_fraction(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333), log_distribute=True, compute_dtype="float32"):
    """https://github.com/tensorflow/models/blob/master/official/vision/image_classification/preprocessing.py
    RandomResizedCrop related function.

    For hh_crop = height, max_ww_crop = height * ratio[1], max_area_crop_1 = height * height * ratio[1]
    For ww_crop = width, max_hh_crop = width / ratio[0], max_area_crop_2 = width * width / ratio[0]
    ==> scale_max < min(max_area_crop_1, max_area_crop_2, scale[1])

    As target_area selected:
    For ww_crop = width, max_aspect_ratio = width / (target_area / width) = width * width / target_area
    For hh_crop = height, min_aspect_ratio = (target_area / height) / height = target_area / (height * height)
    ==> ratio in range (min_aspect_ratio, max_aspect_ratio)

    Result:
    ww_crop * hh_crop = target_area
    ww_crop / hh_crop = aspect_ratio
    ==> ww_crop = int(round(math.sqrt(target_area * aspect_ratio)))
        hh_crop = int(round(math.sqrt(target_area / aspect_ratio)))

    As outputs are converted int, for running 1e5 times, results are not exactly in scale and ratio range:
    >>> from keras_cv_attention_models.imagenet import data
    >>> aa = np.array([data.random_crop_fraction(size=(100, 100), ratio=(0.75, 4./3)) for _ in range(100000)])
    >>> hhs, wws = aa[:, 0], aa[:, 1]
    >>> print("Scale range:", ((hhs * wws).min() / 1e4, (hhs * wws).max() / 1e4))
    # Scale range: (0.075, 0.9801)
    >>> print("Ratio range:", ((wws / hhs).min(), (wws / hhs).max()))
    # Ratio range: (0.7272727272727273, 1.375)

    >>> fig, axes = plt.subplots(4, 1, figsize=(6, 8))
    >>> pp = {
    >>>     "ratio distribute": wws / hhs,
    >>>     "scale distribute": wws * hhs / 1e4,
    >>>     "height distribute": hhs,
    >>>     "width distribute": wws,
    >>> }
    >>> for ax, kk in zip(axes, pp.keys()):
    >>>     _ = ax.hist(pp[kk], bins=1000, label=kk)
    >>>     ax.set_title(kk)
    >>> fig.tight_layout()

    Args:
      size (tuple of int): input image shape. `area = size[0] * size[1]`.
      scale (tuple of float): scale range of the cropped image. target_area in range `(scale[0] * area, sacle[1] * area)`.
      ratio (tuple of float): aspect ratio range of the cropped image. cropped `width / height`  in range `(ratio[0], ratio[1])`.

    Returns: cropped size `hh_crop, ww_crop`.
    """
    height, width = tf.cast(size[0], dtype=compute_dtype), tf.cast(size[1], dtype=compute_dtype)
    area = height * width
    scale_max = tf.minimum(tf.minimum(height * height * ratio[1] / area, width * width / ratio[0] / area), scale[1])
    target_area = tf.random.uniform((), scale[0], scale_max, dtype=compute_dtype) * area

    ratio_min = tf.maximum(target_area / (height * height), ratio[0])
    ratio_max = tf.minimum(width * width / target_area, ratio[1])
    if log_distribute:  # More likely to select a smaller value
        log_min, log_max = tf.math.log(ratio_min), tf.math.log(ratio_max)
        aspect_ratio = tf.random.uniform((), log_min, log_max, dtype=compute_dtype)
        aspect_ratio = tf.math.exp(aspect_ratio)
    else:
        aspect_ratio = tf.random.uniform((), ratio_min, ratio_max, dtype=compute_dtype)

    ww_crop = tf.cast(tf.math.floor(tf.sqrt(target_area * aspect_ratio)), "int32")
    hh_crop = tf.cast(tf.math.floor(tf.sqrt(target_area / aspect_ratio)), "int32")
    # tf.print(">>>> height, width, hh_crop, ww_crop, hh_fraction_min, hh_fraction_max:", height, width, hh_crop, ww_crop, hh_fraction_min, hh_fraction_max)
    # return hh_crop, ww_crop, target_area, hh_fraction_min, hh_fraction_max, hh_fraction
    return hh_crop, ww_crop
    # return hh_fraction, target_area / hh_fraction # float value will stay in scale and ratio range exactly


# Not using
def random_crop_fraction_timm(image, scale=(0.08, 1.0), ratio=(0.75, 1.3333333), compute_dtype="float32"):
    size = tf.shape(image)
    height, width = tf.cast(size[0], dtype=compute_dtype), tf.cast(size[1], dtype=compute_dtype)
    area = height * width
    in_ratio = width / height

    target_areas = tf.random.uniform((10,), scale[0], scale[1]) * area
    log_min, log_max = tf.math.log(ratio[0]), tf.math.log(ratio[1])
    aspect_ratios = tf.random.uniform((10,), log_min, log_max, dtype=compute_dtype)
    aspect_ratios = tf.math.exp(aspect_ratios)

    ww_crops, hh_crops = tf.sqrt(target_areas * aspect_ratios), tf.sqrt(target_areas / aspect_ratios)
    pick = tf.argmax(tf.logical_and(hh_crops <= height, ww_crops <= width))
    hh_crop = tf.cast(tf.math.floor(hh_crops[pick]), "int32")
    ww_crop = tf.cast(tf.math.floor(ww_crops[pick]), "int32")
    # return hh_crop, ww_crop
    return tf.cond(
        tf.logical_and(hh_crop <= size[0], ww_crop <= size[1]),
        lambda: tf.image.random_crop(image, (hh_crop, ww_crop, 3)),
        lambda: tf.image.central_crop(image, tf.minimum(tf.minimum(ratio[1] / in_ratio, in_ratio * ratio[0]), scale[1])),
    )


def random_erasing_per_pixel(image, num_layers=1, scale=(0.02, 0.33333333), ratio=(0.3, 3.3333333), probability=0.5):
    """ https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py """
    if tf.random.uniform(()) > probability:
        return image

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.120003, 57.375]
    height, width, _ = image.shape
    for _ in range(num_layers):
        hh, ww = random_crop_fraction((height, width), scale=scale, ratio=ratio)
        hh_ss = tf.random.uniform((), 0, height - hh, dtype="int32")
        ww_ss = tf.random.uniform((), 0, width - ww, dtype="int32")
        mask = tf.random.normal([hh, ww, 3], mean=mean, stddev=std)
        mask = tf.clip_by_value(mask, 0.0, 255.0)  # value in [0, 255]
        aa = tf.concat([image[:hh_ss, ww_ss : ww_ss + ww], mask, image[hh_ss + hh :, ww_ss : ww_ss + ww]], axis=0)
        image = tf.concat([image[:, :ww_ss], aa, image[:, ww_ss + ww :]], axis=1)
    return image


class RandomProcessImage:
    def __init__(
        self,
        target_shape=(300, 300),
        central_crop=1.0,
        random_crop_min=1.0,
        resize_method="bilinear",
        resize_antialias=False,
        random_erasing_prob=0.0,
        random_erasing_layers=1,
        magnitude=0,
        num_layers=2,
        use_cutout=False,
        use_relative_translate=True,
        use_color_increasing=True,
        **randaug_kwargs,
    ):
        self.magnitude = magnitude
        self.target_shape = target_shape if len(target_shape) == 2 else target_shape[:2]
        self.central_crop, self.random_crop_min = central_crop, random_crop_min
        self.resize_method, self.resize_antialias = resize_method, resize_antialias

        if random_erasing_prob > 0:
            self.random_erasing = lambda img: random_erasing_per_pixel(img, num_layers=random_erasing_layers, probability=random_erasing_prob)
            use_cutout = False
        else:
            self.random_erasing = lambda img: img

        if magnitude > 0:
            from keras_cv_attention_models.imagenet import augment

            # for target_shape = 224, translate_const = 100 and cutout_const = 40
            translate_const = 0.45 if use_relative_translate else min(self.target_shape) * 0.45
            cutout_const = min(self.target_shape) * 0.18
            print(">>>> RandAugment: magnitude = %d, translate_const = %f, cutout_const = %f" % (magnitude, translate_const, cutout_const))

            self.randaug = augment.RandAugment(
                num_layers=num_layers,
                magnitude=magnitude,
                translate_const=translate_const,
                cutout_const=cutout_const,
                use_cutout=use_cutout,
                use_relative_translate=use_relative_translate,
                use_color_increasing=use_color_increasing,
                **randaug_kwargs,
            )

            self.process = self.__train_process__
        elif magnitude == 0:
            self.randaug = lambda img: img
            self.process = self.__train_process__
        else:
            self.process = lambda img: img

    def __train_process__(self, image):
        image = tf.image.random_flip_left_right(image)
        image = self.randaug(image)
        image = self.random_erasing(image)
        return image

    def __call__(self, datapoint):
        image = datapoint["image"]
        if self.random_crop_min > 0 and self.random_crop_min < 1:
            hh, ww = random_crop_fraction(tf.shape(image), scale=(self.random_crop_min, 1.0))
            input_image = tf.image.random_crop(image, (hh, ww, 3))
        else:
            input_image = tf.image.central_crop(image, self.central_crop)
        input_image = tf.image.resize(input_image, self.target_shape, method=self.resize_method, antialias=self.resize_antialias)
        input_image = self.process(input_image)
        input_image = tf.cast(input_image, tf.float32)
        input_image.set_shape([*self.target_shape[:2], 3])

        label = datapoint["label"]
        return input_image, label


def evaluation_process_crop_resize(datapoint, target_shape=(224, 224), central_crop=1.0, resize_method="bilinear", antialias=False):
    image = datapoint["image"]
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    crop_size = tf.cast((central_crop * tf.cast(tf.minimum(height, width), tf.float32)), tf.int32)
    y, x = (height - crop_size) // 2, (width - crop_size) // 2
    image = tf.image.crop_to_bounding_box(image, y, x, crop_size, crop_size)
    image = tf.image.resize(image, target_shape, method=resize_method, antialias=antialias)
    label = datapoint["label"]
    return image, label


# Not using
def evaluation_process_resize_crop(datapoint, target_shape=(224, 224), central_crop=1.0, resize_method="bilinear", antialias=False):
    image = datapoint["image"]
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    min_border = tf.cast(tf.minimum(height, width), tf.float32)
    scale_size = tf.cast(tf.minimum(*target_shape), tf.float32) / central_crop
    hh_scale = tf.cast(tf.floor(tf.cast(height, tf.float32) * scale_size / min_border), tf.int32)
    ww_scale = tf.cast(tf.floor(tf.cast(width, tf.float32) * scale_size / min_border), tf.int32)
    image = tf.image.resize(image, (hh_scale, ww_scale), method=resize_method, antialias=antialias)

    y, x = (hh_scale - target_shape[0]) // 2, (ww_scale - target_shape[1]) // 2
    image = tf.image.crop_to_bounding_box(image, y, x, target_shape[0], target_shape[1])

    label = datapoint["label"]
    return image, label


def sample_beta_distribution(shape, concentration_0=0.4, concentration_1=0.4):
    gamma_1_sample = tf.random.gamma(shape=shape, alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=shape, alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mixup(images, labels, alpha=0.4, min_mix_weight=0):
    """Applies Mixup regularization to a batch of images and labels.

    [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412
    """
    # mix_weight = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    batch_size = tf.shape(images)[0]
    mix_weight = sample_beta_distribution([batch_size], alpha, alpha)
    mix_weight = tf.maximum(mix_weight, 1.0 - mix_weight)

    # For min_mix_weight=0.1, regard values with `> 0.9` as no mixup, this probability is near `1 - alpha`
    # alpha: no_mixup --> {0.1: 0.8128, 0.2: 0.6736, 0.4: 0.4793, 0.6: 0.3521, 0.8: 0.2636, 1.0: 0.2000}
    if min_mix_weight > 0:
        mix_weight = tf.where(mix_weight > 1 - min_mix_weight, tf.ones_like(mix_weight), mix_weight)

    label_mix_weight = tf.cast(tf.expand_dims(mix_weight, -1), "float32")
    img_mix_weight = tf.cast(tf.reshape(mix_weight, [batch_size, 1, 1, 1]), images.dtype)

    labels = tf.cast(labels, "float32")
    # images = images * img_mix_weight + images[::-1] * (1.0 - img_mix_weight)
    # labels = labels * label_mix_weight + labels[::-1] * (1 - label_mix_weight)
    shuffle_index = tf.random.shuffle(tf.range(batch_size))
    images = images * img_mix_weight + tf.gather(images, shuffle_index) * (1.0 - img_mix_weight)
    labels = labels * label_mix_weight + tf.gather(labels, shuffle_index) * (1 - label_mix_weight)
    return images, labels


def get_box(mix_weight, height, width):
    cut_rate_half = tf.math.sqrt(1.0 - mix_weight) / 2
    cut_h_half, cut_w_half = tf.cast(cut_rate_half * float(height), tf.int32), tf.cast(cut_rate_half * float(width), tf.int32)
    cut_h_half, cut_w_half = tf.maximum(1, cut_h_half), tf.maximum(1, cut_w_half)
    # center_y = tf.random.uniform((), minval=cut_h_half, maxval=height - cut_h_half, dtype=tf.int32)
    # center_x = tf.random.uniform((), minval=cut_w_half, maxval=width - cut_w_half, dtype=tf.int32)
    # return center_y - cut_h_half, center_x - cut_w_half, cut_h_half * 2, cut_w_half * 2

    # Can be non-square on border
    center_y = tf.random.uniform((), minval=0, maxval=height, dtype=tf.int32)
    center_x = tf.random.uniform((), minval=0, maxval=width, dtype=tf.int32)
    yl = tf.clip_by_value(center_y - cut_h_half, 0, height)
    yr = tf.clip_by_value(center_y + cut_h_half, 0, height)
    xl = tf.clip_by_value(center_x - cut_w_half, 0, width)
    xr = tf.clip_by_value(center_x + cut_w_half, 0, width)
    return yl, xl, yr - yl, xr - xl


def cutmix(images, labels, alpha=0.5, min_mix_weight=0):
    """
    Copied and modified from https://keras.io/examples/vision/cutmix/

    Example:
    >>> from keras_cv_attention_models.imagenet import data
    >>> import tensorflow_datasets as tfds
    >>> dataset = tfds.load('cifar10', split='train').batch(16)
    >>> dd = dataset.as_numpy_iterator().next()
    >>> images, labels = dd['image'], tf.one_hot(dd['label'], depth=10)
    >>> aa, bb = data.cutmix(images, labels)
    >>> print(bb.numpy()[bb.numpy() != 0])
    >>> plt.imshow(np.hstack(aa))
    """
    # Get a sample from the Beta distribution
    batch_size = tf.shape(images)[0]
    _, hh, ww, _ = images.shape
    mix_weight = sample_beta_distribution((), alpha, alpha)  # same value in batch
    offset_height, offset_width, target_height, target_width = get_box(mix_weight, hh, ww)
    mix_weight = 1.0 - tf.cast(target_height * target_width, "float32") / tf.cast(hh * ww, "float32")
    if mix_weight < min_mix_weight or 1 - mix_weight < min_mix_weight:
        # For input_shape=224, min_mix_weight=0.1, min_height = 224 * sqrt(0.1) = 70.835
        return images, labels

    crops = tf.image.crop_to_bounding_box(images, offset_height, offset_width, target_height, target_width)
    pad_crops = tf.image.pad_to_bounding_box(crops, offset_height, offset_width, hh, ww)

    labels = tf.cast(labels, "float32")
    # images = images - pad_crops + pad_crops[::-1]
    # labels = labels * mix_weight + labels[::-1] * (1.0 - mix_weight)
    shuffle_index = tf.random.shuffle(tf.range(batch_size))
    images = images - pad_crops + tf.gather(pad_crops, shuffle_index)
    labels = labels * mix_weight + tf.gather(labels, shuffle_index) * (1.0 - mix_weight)
    return images, labels


def init_mean_std_by_rescale_mode(rescale_mode):
    if isinstance(rescale_mode, (list, tuple)):  # Specific mean and std
        mean, std = rescale_mode
    elif rescale_mode == "torch":
        mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
        std = tf.constant([0.229, 0.224, 0.225]) * 255.0
    elif rescale_mode == "tf":
        # mean, std = 128.0, 128.0
        mean, std = 127.5, 128.0
    else:
        mean, std = 0, 1  # raw inputs [0, 255]
    return mean, std


def init_dataset(
    data_name="imagenet2012",  # dataset params
    input_shape=(224, 224),
    batch_size=64,
    buffer_size=1000,
    info_only=False,
    mixup_alpha=0,  # mixup / cutmix params
    cutmix_alpha=0,
    rescale_mode="tf",  # rescale mode, ["tf", "torch"], or specific `(mean, std)` like `(128.0, 128.0)`
    eval_central_crop=1.0,  # augment params
    random_crop_min=1.0,
    resize_method="bilinear",  # ["bilinear", "bicubic"]
    resize_antialias=False,
    random_erasing_prob=0.0,
    magnitude=0,
    num_layers=2,
    **augment_kwargs,  # Too many...
):
    """Init dataset by name.
    returns train_dataset, test_dataset, total_images, num_classes, steps_per_epoch
    """
    # print(">>>> Dataset args:", locals())
    try_gcs = True if len(tf.config.list_logical_devices("TPU")) > 0 else False
    dataset, info = tfds.load(data_name, with_info=True, try_gcs=try_gcs)
    num_classes = info.features["label"].num_classes
    total_images = info.splits["train"].num_examples
    steps_per_epoch = int(tf.math.ceil(total_images / float(batch_size)))
    if info_only:
        return total_images, num_classes, steps_per_epoch

    AUTOTUNE = tf.data.AUTOTUNE
    train_process = RandomProcessImage(
        target_shape=input_shape,
        central_crop=1.0,
        random_crop_min=random_crop_min,
        resize_method=resize_method,
        resize_antialias=resize_antialias,
        random_erasing_prob=random_erasing_prob,
        magnitude=magnitude,
        num_layers=num_layers,
        **augment_kwargs,
    )
    train_dataset = dataset["train"].shuffle(buffer_size).map(lambda xx: train_process(xx), num_parallel_calls=AUTOTUNE)

    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    # rescaling = lambda xx: (tf.clip_by_value(xx, 0, 255) - mean) / std
    rescaling = lambda xx: (xx - mean) / std
    as_one_hot = lambda yy: tf.one_hot(yy, num_classes)

    train_dataset = train_dataset.batch(batch_size)
    if mixup_alpha > 0 and mixup_alpha <= 1 and cutmix_alpha > 0 and cutmix_alpha <= 1:
        print(">>>> Both mixup_alpha and cutmix_alpha provided: mixup_alpha = {}, cutmix_alpha = {}".format(mixup_alpha, cutmix_alpha))
        mixup_cutmix = lambda xx, yy: tf.cond(
            tf.random.uniform(()) > 0.5,  # switch_prob = 0.5
            lambda: mixup(rescaling(xx), as_one_hot(yy), alpha=mixup_alpha),
            lambda: cutmix(rescaling(xx), as_one_hot(yy), alpha=cutmix_alpha),
        )
        train_dataset = train_dataset.map(mixup_cutmix, num_parallel_calls=AUTOTUNE)
    elif mixup_alpha > 0 and mixup_alpha <= 1:
        print(">>>> mixup_alpha provided:", mixup_alpha)
        train_dataset = train_dataset.map(lambda xx, yy: mixup(rescaling(xx), as_one_hot(yy), alpha=mixup_alpha), num_parallel_calls=AUTOTUNE)
    elif cutmix_alpha > 0 and cutmix_alpha <= 1:
        print(">>>> cutmix_alpha provided:", cutmix_alpha)
        train_dataset = train_dataset.map(lambda xx, yy: cutmix(rescaling(xx), as_one_hot(yy), alpha=cutmix_alpha), num_parallel_calls=AUTOTUNE)
    else:
        train_dataset = train_dataset.map(lambda xx, yy: (rescaling(xx), as_one_hot(yy)), num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    """ Test dataset """
    test_process = lambda xx: evaluation_process_crop_resize(xx, input_shape[:2], eval_central_crop, resize_method, resize_antialias)
    # test_process = lambda xx: evaluation_process_resize_crop(xx, input_shape[:2], eval_central_crop, resize_method, resize_antialias)  # timm
    if "validation" in dataset:
        test_dataset = dataset["validation"].map(test_process)
    elif "test" in dataset:
        test_dataset = dataset["test"].map(test_process)
    test_dataset = test_dataset.batch(batch_size).map(lambda xx, yy: (rescaling(xx), as_one_hot(yy)))
    return train_dataset, test_dataset, total_images, num_classes, steps_per_epoch
