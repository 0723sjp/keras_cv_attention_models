import numpy as np
import tensorflow as tf


def __compute_loss__(feature_extractor, input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def __gradient_ascent_step__(feature_extractor, img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = __compute_loss__(feature_extractor, img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def __initialize_image__(img_width, img_height):
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def __deprocess_image__(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def __get_cols_rows__(total, rows=-1):
    if rows == -1 and total < 8:
        rows = 1  # for total in [1, 7], plot 1 row only
    elif rows == -1:
        rr = int(np.floor(np.sqrt(total)))
        for ii in range(1, rr + 1)[::-1]:
            if total % ii == 0:
                rows = ii
                break
    cols = total // rows
    return cols, rows


def stack_and_plot_images(images, margin=5, margin_value=0, rows=-1, ax=None, base_size=3):
    """ Stack and plot a list of images. Returns ax, stacked_images """
    import matplotlib.pyplot as plt

    cols, rows = __get_cols_rows__(len(images), rows)
    images = images[: rows * cols]
    # print(">>>> rows:", rows, ", cols:", cols, ", total:", len(images))

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(base_size * cols, base_size * rows))

    if margin > 0:
        channel = images[0].shape[-1]
        ww_margin = np.zeros([images[0].shape[0], margin, channel], dtype=images[0].dtype) + margin_value
        ww_margined_images = [np.hstack([ii, ww_margin]) for ii in images]
        hstacked_images = [np.hstack(ww_margined_images[ii : ii + cols]) for ii in range(0, len(ww_margined_images), cols)]

        hh_margin = np.zeros([margin, hstacked_images[0].shape[1], channel], dtype=hstacked_images[0].dtype) + margin_value
        hh_margined_images = [np.vstack([ii, hh_margin]) for ii in hstacked_images]
        vstacked_images = np.vstack(hh_margined_images)

        stacked_images = vstacked_images[:-margin, :-margin]
    else:
        stacked_images = np.vstack([np.hstack(images[ii * cols : (rr + 1) * cols]) for ii in range(rows)])

    ax.imshow(stacked_images)
    ax.set_axis_off()
    ax.grid(False)
    plt.tight_layout()
    return ax, stacked_images


def visualize_filters(model, layer_name, filter_index_list=[0], input_shape=None, iterations=30, learning_rate=10.0, base_size=3):
    # Set up a model that returns the activation values for our target layer
    layer = model.get_layer(name=layer_name)
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
    input_shape = model.input_shape[1:-1] if input_shape is None else input_shape[:2]
    print(">>>> Total filters for layer {}: {}, input_shape: {}".format(layer_name, layer.output_shape[-1], input_shape))

    # We run gradient ascent for [iterations] steps
    losses, filter_images = [], []
    for filter_index in filter_index_list:
        print("Processing filter %d" % (filter_index,))
        image = __initialize_image__(input_shape[0], input_shape[1])
        for iteration in range(iterations):
            loss, image = __gradient_ascent_step__(feature_extractor, image, filter_index, learning_rate)
        # Decode the resulting input image
        image = __deprocess_image__(image[0].numpy())
        losses.append(loss.numpy())
        filter_images.append(image)

    ax, _ = stack_and_plot_images(filter_images, base_size=base_size)
    return losses, np.stack(filter_images), ax


def make_gradcam_heatmap(model, img_array, layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(model.inputs[0], [model.get_layer(layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=list(range(0, len(grads.shape) - 1)))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    if len(heatmap.shape) > 2:
        heatmap = tf.reduce_mean(heatmap, list(range(0, len(heatmap.shape) - 2)))

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy().astype("float32"), preds.numpy()


def make_and_apply_gradcam_heatmap(model, image, layer_name, rescale_mode="tf", pred_index=None, alpha=0.4):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    processed_image = tf.expand_dims(tf.image.resize(image, model.input_shape[1:-1]), 0)
    processed_image = tf.keras.applications.imagenet_utils.preprocess_input(processed_image, mode=rescale_mode)
    heatmap, preds = make_gradcam_heatmap(model, processed_image, layer_name, pred_index=pred_index)

    # Use jet colormap to colorize heatmap. Use RGB values of the colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(tf.range(256))[:, :3]
    jet_heatmap = jet_colors[tf.cast(heatmap * 255, "uint8").numpy()]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.image.resize(jet_heatmap, (image.shape[:2]))  # [0, 1]

    # Superimpose the heatmap on original image
    image = image.astype("float32") / 255 if image.max() > 127 else image
    superimposed_img = (jet_heatmap * alpha + image).numpy()
    superimposed_img /= superimposed_img.max()

    print(">>>> Prediction:", tf.keras.applications.imagenet_utils.decode_predictions(preds)[0])
    print(">>>> Top 5 prediction indexes:", np.argsort(preds[0])[-5:])
    fig = plt.figure()
    plt.imshow(superimposed_img)
    plt.axis("off")
    plt.tight_layout()
    return superimposed_img, heatmap, preds


def matmul_prod(aa):
    vv = aa[0]
    for ii in aa[1:]:
        vv = np.matmul(vv, ii)
    return vv


def apply_mask_2_image(image, mask):
    if len(mask.shape) == 1:
        width = height = int(np.sqrt(mask.shape[0]))
        mask = mask[-width * height :]
    else:
        height, width = mask.shape[:2]
    mask = mask.reshape(width, height, 1)
    mask = tf.image.resize(mask / mask.max(), image.shape[:2]).numpy()
    return (mask * image).astype("uint8")


def clip_max_value_matrix(dd, axis=0):
    # print("Before:", dd.max())
    for ii in range(dd.shape[axis]):
        if axis == 0:
            max_idx = np.argmax(dd[ii])
            dd[ii, max_idx] = dd[ii].min()
            dd[ii, max_idx] = dd[ii].max()
        else:
            max_idx = np.argmax(dd[:, ii])
            dd[max_idx, ii] = dd[:, ii].min()
            dd[max_idx, ii] = dd[:, ii].max()
    # print("After:", dd.max())
    return dd


def down_sample_matrix_axis_0(dd, target, method="avg"):
    if dd.shape[0] == target:
        return dd

    rate = int(np.sqrt(dd.shape[0] // target))
    hh = ww = int(np.sqrt(dd.shape[0]))
    dd = dd.reshape(1, hh, ww, -1)
    if "avg" in method.lower():
        dd = tf.nn.avg_pool(dd, rate, rate, "VALID").numpy()
    else:
        dd = tf.nn.max_pool(dd, rate, rate, "VALID").numpy()
    dd = dd.reshape(-1, dd.shape[-1])
    return dd


def plot_attention_score_maps(model, image, rescale_mode="tf", attn_type="auto", rows=-1, base_size=3):
    import matplotlib.pyplot as plt

    if isinstance(model, tf.keras.models.Model):
        imm_inputs = tf.keras.applications.imagenet_utils.preprocess_input(image, mode=rescale_mode)
        imm_inputs = tf.expand_dims(tf.image.resize(imm_inputs, model.input_shape[1:3]), 0)
        try:
            pred = model(imm_inputs).numpy()
            if model.layers[-1].activation.__name__ != "softmax":
                pred = tf.nn.softmax(pred).numpy()  # If classifier activation is not softmax
            print(">>>> Prediction:", tf.keras.applications.imagenet_utils.decode_predictions(pred)[0])
        except:
            pass
        bb = tf.keras.models.Model(model.inputs[0], [ii.output for ii in model.layers if ii.name.endswith("attention_scores")])
        attn_scores = bb(imm_inputs)
        layer_name_title = "\nLayer name: {} --> {}".format(bb.output_names[-1], bb.output_names[0])
    else:
        attn_scores = model
        layer_name_title = ""
        assert attn_type != "auto"

    attn_type = attn_type.lower()
    check_type_is = lambda tt: (tt in model.name.lower()) if attn_type == "auto" else (attn_type.startswith(tt))
    if check_type_is("beit"):
        # beit attn_score [batch, num_heads, cls_token + hh * ww, cls_token + hh * ww]
        print(">>>> Attention type: beit")
        mask = [np.array(ii)[0].mean(0) + np.eye(ii.shape[-1]) for ii in attn_scores][::-1]
        mask = [(ii / ii.sum()) for ii in mask]
        cum_mask = [matmul_prod(mask[: ii + 1])[0] for ii in range(len(mask))]
        mask = [ii[0] for ii in mask]
    elif check_type_is("levit"):
        # levit attn_score [batch, num_heads, q_blocks, k_blocks]
        print(">>>> Attention type: levit")
        mask = [np.array(ii)[0].mean(0) for ii in attn_scores][::-1]
        cum_mask = [matmul_prod(mask[: ii + 1]).mean(0) for ii in range(len(mask))]
        mask = [ii.mean(0) for ii in mask]
    elif check_type_is("bot") or check_type_is("coatnet"):
        # bot attn_score [batch, num_heads, hh * ww, hh * ww]
        print(">>>> Attention type: bot / coatnet")
        mask = [np.array(ii)[0].mean((0)) for ii in attn_scores if len(ii.shape) == 4][::-1]
        mask = [clip_max_value_matrix(ii) for ii in mask]  # Or it will be too dark.
        method = "max" if check_type_is("coatnet") else "avg"
        cum_mask = [mask[0]] + [down_sample_matrix_axis_0(mask[ii], mask[ii - 1].shape[1], method) for ii in range(1, len(mask))]
        cum_mask = [matmul_prod(cum_mask[: ii + 1]).mean(0) for ii in range(len(cum_mask))]
        mask = [ii.mean(0) for ii in mask]
    elif check_type_is("halo"):
        # halo attn_score [batch, num_heads, hh, ww, query_block * query_block, kv_kernel * kv_kernel]
        print(">>>> Attention type: halo")
        from einops import rearrange
        from keras_cv_attention_models.attention_layers import CompatibleExtractPatches

        mask = [np.array(ii)[0].mean(0) for ii in attn_scores if len(ii.shape) == 6][::-1]

        qqs = [int(np.sqrt(ii.shape[2])) for ii in mask]  # query_kernel
        vvs = [int(np.sqrt(ii.shape[3])) for ii in mask]  # kv_kernel
        hhs = [(jj - ii) // 2 for ii, jj in zip(qqs, vvs)]  # halo_size
        tt = [rearrange(ii, "hh ww (hb wb) cc -> (hh hb) (ww wb) cc", hb=qq, wb=qq) for ii, qq in zip(mask, qqs)]
        tt = [tf.expand_dims(tf.pad(ii, [[hh, hh], [hh, hh], [0, 0]]), 0) for ii, hh in zip(tt, hhs)]
        tt = [CompatibleExtractPatches(vv, qq, padding="VALID", compressed=False)(ii).numpy()[0] for ii, vv, qq in zip(tt, vvs, qqs)]
        tt = [rearrange(ii, "hh ww hb wb cc -> hh ww (hb wb) cc").mean((0, 1)) for ii in tt]
        # tt = [tf.reduce_max(rearrange(ii, "hh ww hb wb cc -> hh ww (hb wb) cc"), axis=(0, 1)).numpy() for ii in tt]
        cum_mask = [matmul_prod(tt[: ii + 1]).mean(0) for ii in range(len(tt))]
        mask = [ii.mean((0, 1, 2)) for ii in mask]
    else:
        print(">>>> Attention type: cot / volo / unknown")
        # cot attn_score [batch, 1, 1, filters, randix]
        # volo attn_score [batch, hh, ww, num_heads, kernel_size * kernel_size, kernel_size * kernel_size]
        print("[{}] still don't know how...".format(attn_type))
        return None, None

    masked_image = [apply_mask_2_image(image, ii) for ii in mask]
    cum_masked_image = [apply_mask_2_image(image, ii) for ii in cum_mask]

    cols, rows = __get_cols_rows__(len(mask), rows)
    fig, axes = plt.subplots(2, 1, figsize=(base_size * cols, base_size * rows * 2))
    stack_and_plot_images(masked_image, margin=5, rows=rows, ax=axes[0])
    axes[0].set_title("Attention scores: attn_scores[{}] --> attn_scores[0]".format(len(mask)) + layer_name_title)
    stack_and_plot_images(cum_masked_image, margin=5, rows=rows, ax=axes[1])
    axes[1].set_title("Accumulated attention scores: attn_scores[{}:] --> attn_scores[0:]".format(len(mask) - 1) + layer_name_title)
    fig.tight_layout()
    return mask, cum_mask, fig
