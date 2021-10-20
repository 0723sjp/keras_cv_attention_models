import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img


def random_crop_fraction(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333), log_distribute=True):
    """https://github.com/tensorflow/models/blob/master/official/vision/image_classification/preprocessing.py
    RandomResizedCrop related function.

    For hh_crop = height, max_ww_crop = height * ratio[1], max_area_crop_1 = height * height * ratio[1]
    For ww_crop = width, max_hh_crop = width / ratio[0], max_area_crop_2 = width * width / ratio[0]
    ==> scale_max < min(max_area_crop_1, max_area_crop_2, scale[1])

    hh_crop * ww_crop = crop_area = crop_fraction * area, crop_fraction in scale=(scale[0], scale_max)
    ww_crop / hh_crop > ratio[0]
    ww_crop / hh_crop < ratio[1]
    ==> hh_crop > sqrt(crop_area / ratio[1])
        hh_crop < sqrt(crop_area / ratio[0])

    As outputs are converted int, for running 1e5 times, results are not exactly in scale and ratio range:
    >>> from keras_cv_attention_models.imagenet import data
    >>> aa = np.array([data.random_crop_fraction(size=(100, 100), ratio=(0.75, 4./3)) for _ in range(100000)])
    >>> print("Scale range:", ((aa[:, 0] * aa[:, 1]).min() / 1e4, (aa[:, 0] * aa[:, 1]).max() / 1e4))
    # Scale range: (0.075, 0.9801)
    >>> print("Ratio range:", ((aa[:, 0] / aa[:, 1]).min(), (aa[:, 0] / aa[:, 1]).max()))
    # Ratio range: (0.7272727272727273, 1.375)

    >>> fig, axes = plt.subplots(4, 1, figsize=(6, 8))
    >>> pp = {
    >>>     "ratio distribute": aa[:, 1] / aa[:, 0],
    >>>     "scale distribute": aa[:, 1] * aa[:, 0] / 1e4,
    >>>     "height distribute": aa[:, 0],
    >>>     "width distribute": aa[:, 1],
    >>> }
    >>> for ax, kk in zip(axes, pp.keys()):
    >>>     _ = ax.hist(pp[kk], bins=1000, label=kk)
    >>>     ax.set_title(kk)
    >>> fig.tight_layout()

    Args:
      size (tuple of int): input image shape. `area = size[0] * size[1]`.
      scale (tuple of float): scale range of the cropped image. crop_area in range `(scale[0] * area, sacle[1] * area)`.
      ratio (tuple of float): aspect ratio range of the cropped image. cropped `width / height`  in range `(ratio[0], ratio[1])`.

    Returns: cropped size `hh_crop, ww_crop`.
    """
    height, width = tf.cast(size[0], "float32"), tf.cast(size[1], "float32")
    area = height * width
    scale_max = tf.minimum(tf.minimum(height * height * ratio[1] / area, width * width / ratio[0] / area), scale[1])
    crop_area = tf.random.uniform((), scale[0], scale_max) * area
    hh_fraction_min = tf.maximum(crop_area / width, tf.sqrt(crop_area / ratio[1]))
    hh_fraction_max = tf.minimum(tf.cast(height, "float32"), tf.sqrt(crop_area / ratio[0]))
    if log_distribute:  # More likely to select a smaller value
        log_min, log_max = tf.math.log(hh_fraction_min), tf.math.log(hh_fraction_max)
        hh_fraction = tf.random.uniform((), log_min, log_max)
        hh_fraction = tf.math.exp(hh_fraction)
    else:
        hh_fraction = tf.random.uniform((), hh_fraction_min, hh_fraction_max)
    # hh_crop, ww_crop = tf.cast(tf.math.ceil(hh_fraction), "int32"), tf.cast(tf.math.ceil(crop_area / hh_fraction), "int32")
    hh_crop, ww_crop = tf.cast(tf.math.floor(hh_fraction), "int32"), tf.cast(tf.math.floor(crop_area / hh_fraction), "int32")
    # tf.print(">>>> height, width, hh_crop, ww_crop, hh_fraction_min, hh_fraction_max:", height, width, hh_crop, ww_crop, hh_fraction_min, hh_fraction_max)
    # return hh_crop, ww_crop, crop_area, hh_fraction_min, hh_fraction_max, hh_fraction
    return hh_crop, ww_crop
    # return hh_fraction, crop_area / hh_fraction # float value will stay in scale and ratio range exactly
    # return tf.minimum(hh_crop, size[0] - 1), tf.minimum(ww_crop, size[1] - 1)


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
        self.central_crop, self.random_crop_min, self.resize_method = central_crop, random_crop_min, resize_method

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
            # tf.print(tf.shape(image), hh, ww)
            input_image = tf.image.random_crop(image, (hh, ww, 3))
        else:
            input_image = tf.image.central_crop(image, self.central_crop)
        input_image = tf.image.resize(input_image, self.target_shape, method=self.resize_method)
        input_image = self.process(input_image)
        input_image = tf.cast(input_image, tf.float32)
        input_image.set_shape([*self.target_shape[:2], 3])

        label = datapoint["label"]
        return input_image, label


def sample_beta_distribution(size, concentration_0=0.4, concentration_1=0.4):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mixup(image, label, alpha=0.4):
    """Applies Mixup regularization to a batch of images and labels.

    [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412
    """
    # mix_weight = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    batch_size = tf.shape(image)[0]
    mix_weight = sample_beta_distribution(batch_size, alpha, alpha)
    mix_weight = tf.maximum(mix_weight, 1.0 - mix_weight)

    # Regard values with `> 0.9` as no mixup, this probability is near `1 - alpha`
    # alpha: no_mixup --> {0.2: 0.6714, 0.4: 0.47885, 0.6: 0.35132, 0.8: 0.26354, 1.0: 0.19931}
    mix_weight = tf.where(mix_weight > 0.9, tf.ones_like(mix_weight), mix_weight)

    label_mix_weight = tf.cast(tf.expand_dims(mix_weight, -1), "float32")
    img_mix_weight = tf.cast(tf.reshape(mix_weight, [batch_size, 1, 1, 1]), image.dtype)

    shuffle_index = tf.random.shuffle(tf.range(batch_size))
    image = image * img_mix_weight + tf.gather(image, shuffle_index) * (1.0 - img_mix_weight)
    label = tf.cast(label, "float32")
    label = label * label_mix_weight + tf.gather(label, shuffle_index) * (1 - label_mix_weight)
    return image, label


def get_box(mix_weight, height, width):
    cut_rate_half = tf.math.sqrt(1.0 - mix_weight) / 2
    cut_h_half, cut_w_half = tf.cast(cut_rate_half * float(height), tf.int32), tf.cast(cut_rate_half * float(width), tf.int32)
    center_y = tf.random.uniform((1,), minval=cut_h_half, maxval=height - cut_h_half, dtype=tf.int32)[0]
    center_x = tf.random.uniform((1,), minval=cut_w_half, maxval=width - cut_w_half, dtype=tf.int32)[0]
    return center_y - cut_h_half, center_x - cut_w_half, cut_h_half * 2, cut_w_half * 2


def cutmix(images, labels, alpha=0.5, min_mix_weight=0.01):
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
    mix_weight = sample_beta_distribution(1, alpha, alpha)[0]  # same value in batch
    if mix_weight < min_mix_weight or 1 - mix_weight < min_mix_weight:
        # For input_shape=224, min_mix_weight=0.01, min_height = 224 * 0.1 = 22.4
        return images, labels

    offset_height, offset_width, target_height, target_width = get_box(mix_weight, hh, ww)
    crops = tf.image.crop_to_bounding_box(images, offset_height, offset_width, target_height, target_width)
    pad_crops = tf.image.pad_to_bounding_box(crops, offset_height, offset_width, hh, ww)

    shuffle_index = tf.random.shuffle(tf.range(batch_size))
    images = images - pad_crops + tf.gather(pad_crops, shuffle_index)
    labels = tf.cast(labels, "float32")
    label_mix_weight = tf.cast(tf.expand_dims(mix_weight, -1), "float32")
    labels = labels * label_mix_weight + tf.gather(labels, shuffle_index) * (1 - label_mix_weight)
    return images, labels


def init_dataset(
    data_name="imagenet2012",  # dataset params
    input_shape=(224, 224),
    batch_size=64,
    buffer_size=1000,
    info_only=False,
    mixup_alpha=0,  # mixup / cutmix params
    cutmix_alpha=0,
    rescale_mode="tf",  # rescale mode, ["tf", "torch"]
    central_crop=1.0,  # augment params
    random_crop_min=1.0,
    resize_method="bilinear",  # ["bilinear", "bicubic"]
    random_erasing_prob=0.0,
    magnitude=0,
    num_layers=2,
    **augment_kwargs,  # Too many...
):
    """Init dataset by name.
    returns train_dataset, test_dataset, total_images, num_classes, steps_per_epoch
    """
    dataset, info = tfds.load(data_name, with_info=True)
    num_classes = info.features["label"].num_classes
    total_images = info.splits["train"].num_examples
    steps_per_epoch = int(tf.math.ceil(total_images / float(batch_size)))
    if info_only:
        return total_images, num_classes, steps_per_epoch

    AUTOTUNE = tf.data.AUTOTUNE
    train_process = RandomProcessImage(
        target_shape=input_shape,
        central_crop=central_crop,
        random_crop_min=random_crop_min,
        resize_method=resize_method,
        random_erasing_prob=random_erasing_prob,
        magnitude=magnitude,
        num_layers=num_layers,
        **augment_kwargs,
    )
    train = dataset["train"].map(lambda xx: train_process(xx), num_parallel_calls=AUTOTUNE)
    test_process = RandomProcessImage(input_shape, magnitude=-1, central_crop=central_crop, random_crop_min=1.0, resize_method=resize_method)
    if "validation" in dataset:
        test = dataset["validation"].map(lambda xx: test_process(xx))
    elif "test" in dataset:
        test = dataset["test"].map(lambda xx: test_process(xx))

    if rescale_mode == "torch":
        mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
        std = tf.constant([0.229, 0.224, 0.225]) * 255.0
        rescaling = lambda xx: (xx - mean) / std
    else:
        rescaling = lambda xx: (xx - 127.5) * 0.0078125

    as_one_hot = lambda yy: tf.one_hot(yy, num_classes)
    train_dataset = train.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test.batch(batch_size).map(lambda xx, yy: (rescaling(xx), as_one_hot(yy)))

    if mixup_alpha > 0 and mixup_alpha <= 1 and cutmix_alpha > 0 and cutmix_alpha <= 1:
        print(">>>> Both mixup_alpha and cutmix_alpha provided: mixup_alpha = {}, cutmix_alpha = {}".format(mixup_alpha, cutmix_alpha))
        mixup_cutmix = lambda xx, yy: tf.cond(
            tf.random.uniform(()) > 0.5,  # switch_prob = 0.5
            lambda: mixup(rescaling(xx), as_one_hot(yy), alpha=mixup_alpha),
            lambda: cutmix(rescaling(xx), as_one_hot(yy), alpha=cutmix_alpha),
        )
        train_dataset = train_dataset.map(mixup_cutmix)
    elif mixup_alpha > 0 and mixup_alpha <= 1:
        print(">>>> mixup_alpha provided:", mixup_alpha)
        train_dataset = train_dataset.map(lambda xx, yy: mixup(rescaling(xx), as_one_hot(yy), alpha=mixup_alpha))
    elif cutmix_alpha > 0 and cutmix_alpha <= 1:
        print(">>>> cutmix_alpha provided:", cutmix_alpha)
        train_dataset = train_dataset.map(lambda xx, yy: cutmix(rescaling(xx), as_one_hot(yy), alpha=cutmix_alpha))
    else:
        train_dataset = train_dataset.map(lambda xx, yy: (rescaling(xx), as_one_hot(yy)))
    return train_dataset, test_dataset, total_images, num_classes, steps_per_epoch
