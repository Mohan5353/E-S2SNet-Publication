import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def get_weight(shape, gain=np.sqrt(2)):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)
    w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
    return w

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])

def conv2d_bias(x, fmaps, kernel, gain=np.sqrt(2)):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain)
    w = tf.cast(w, x.dtype)
    x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "SYMMETRIC")
    return apply_bias(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW'))

def maxpool2d(x, k=2):
    ksize = [1, 1, k, k]
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

def conv_lr(name, x, fmaps, p=0.7):
    with tf.variable_scope(name):
        x = tf.nn.dropout(x, p)
        return tf.nn.leaky_relu(conv2d_bias(x, fmaps, 3), alpha=0.1)

def conv(name, x, fmaps, p):
    with tf.variable_scope(name):
        x = tf.nn.dropout(x, p)
        return tf.nn.sigmoid(conv2d_bias(x, fmaps, 3, gain=1.0))

def gated_conv_lr(name, x, fmaps):
    with tf.variable_scope(name):
        with tf.variable_scope('feature'):
            feature = conv2d_bias(x, fmaps, 3)
        with tf.variable_scope('gating'):
            gating = tf.nn.sigmoid(conv2d_bias(x, fmaps, 3, gain=1.0))
        x_out = feature * gating
        return tf.nn.leaky_relu(x_out, alpha=0.1)

def gated_conv_unet(x, channel=3, width=256, height=256, p=0.7, **_kwargs):
    x.set_shape([None, channel, height, width])
    skips = [x]

    n = x
    n = gated_conv_lr('enc_conv0', n, 48)
    n = gated_conv_lr('enc_conv1', n, 48)
    n = maxpool2d(n)
    skips.append(n)

    n = gated_conv_lr('enc_conv2', n, 48)
    n = maxpool2d(n)
    skips.append(n)

    n = gated_conv_lr('enc_conv3', n, 48)
    n = maxpool2d(n)
    skips.append(n)

    n = gated_conv_lr('enc_conv4', n, 48)
    n = maxpool2d(n)
    skips.append(n)

    n = gated_conv_lr('enc_conv5', n, 48)
    n = maxpool2d(n)
    n = gated_conv_lr('enc_conv6', n, 48)

    # -----------------------------------------------
    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv5', n, 96, p=p)
    n = conv_lr('dec_conv5b', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv4', n, 96, p=p)
    n = conv_lr('dec_conv4b', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv3', n, 96, p=p)
    f3 = conv_lr('dec_conv3b', n, 96, p=p)

    n = upscale2d(f3)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv2', n, 96, p=p)
    f2 = conv_lr('dec_conv2b', n, 96, p=p)

    n = upscale2d(f2)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv1a', n, 64, p=p)
    f1 = conv_lr('dec_conv1b', n, 32, p=p)
    n = conv('dec_conv1', f1, channel, p=p)

    feature_maps = {'dec_conv3b': f3, 'dec_conv2b': f2, 'dec_conv1b': f1}
    return n, feature_maps

def concat(x, y):
    bs1, c1, h1, w1 = x.shape.as_list()
    bs2, c2, h2, w2 = y.shape.as_list()
    x = tf.image.crop_to_bounding_box(tf.transpose(x, [0, 2, 3, 1]), 0, 0, min(h1, h2), min(w1, w2))
    y = tf.image.crop_to_bounding_box(tf.transpose(y, [0, 2, 3, 1]), 0, 0, min(h1, h2), min(w1, w2))
    return tf.transpose(tf.concat([x, y], axis=3), [0, 3, 1, 2])

def build_denoising_unet(noisy_tensor, keep_prob, is_realnoisy, alpha, beta, mask_prob=0.5, reg_layer='dec_conv2b'):
    # noisy_tensor is NCHW
    _, h, w, c = np.shape(noisy_tensor)
    is_flip_lr = tf.placeholder(tf.int16)
    is_flip_ud = tf.placeholder(tf.int16)
    noisy_tensor = data_arg(noisy_tensor, is_flip_lr, is_flip_ud)
    response = tf.transpose(noisy_tensor, [0, 3, 1, 2])
    
    mask_tensor = tf.ones_like(response)
    mask_tensor = tf.nn.dropout(mask_tensor, mask_prob) * mask_prob
    masked_response = tf.multiply(mask_tensor, response)
    
    if is_realnoisy:
        response = tf.squeeze(tf.random_poisson(25 * response, [1]) / 25, 0)
        masked_response = tf.squeeze(tf.random_poisson(25 * masked_response, [1]) / 25, 0)
        
    combined_input = tf.concat([response, masked_response], axis=0)
    combined_output, combined_features = gated_conv_unet(combined_input, channel=c, width=w, height=h, p=keep_prob)
    
    output_y, output_y_m = tf.split(combined_output, 2, axis=0)
    feature_map_y_m = combined_features[reg_layer][1:2]
    
    output_y = tf.transpose(output_y, [0, 2, 3, 1])
    output_y_m = tf.transpose(output_y_m, [0, 2, 3, 1])
    mask_tensor = tf.transpose(mask_tensor, [0, 2, 3, 1])
    

    data_loss = mask_loss(output_y_m, noisy_tensor, 1. - mask_tensor)
    l_gc = tf.reduce_mean(tf.abs(output_y - output_y_m))
    
    dy_in, dx_in = tf.image.image_gradients(noisy_tensor)
    
    # Use input gradient magnitude to weight flat regions (noise) for energy loss
    mag_in = tf.sqrt(tf.square(dy_in) + tf.square(dx_in) + 1e-8)
    
    # feature_map_y_m is NCHW. E_i,j = log(sum exp) over channels.
    energy_map = tf.reduce_logsumexp(feature_map_y_m, axis=1) # Shape: (1, H', W')
    energy_map = tf.expand_dims(energy_map, axis=-1) # Shape: (1, H', W', 1)
    
    # mag_in is (1, H, W, 3). Reduce to 1 channel and resize to (H', W')
    mag_in_gray = tf.reduce_max(mag_in, axis=-1, keepdims=True)
    energy_shape = tf.shape(energy_map)[1:3]
    mag_in_resized = tf.image.resize(mag_in_gray, energy_shape)
    
    # flat_weight is ~1.0 for flat regions, and ~0.0 for edges/textures
    flat_weight = tf.exp(-10.0 * mag_in_resized)
    l_energy = tf.abs(tf.reduce_mean(flat_weight * (1.0 / (tf.abs(energy_map) + 1e-8))))
    
    # Total loss is data_loss + alpha * l_gc + beta * l_energy (edge loss removed)
    total_loss = data_loss + alpha * l_gc + beta * l_energy
    
    output_y_m = data_arg(output_y_m, is_flip_lr, is_flip_ud)
    
    slice_avg = tf.get_variable('slice_avg', shape=[1, h, w, c], initializer=tf.initializers.zeros())
    avg_op = slice_avg.assign(slice_avg * 0.99 + output_y_m * 0.01)
    our_image = output_y_m

    training_error = total_loss
    tf.summary.scalar('data loss', data_loss)
    tf.summary.scalar('gc loss', l_gc)
    tf.summary.scalar('energy loss', l_energy)
    tf.summary.scalar('total loss', total_loss)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=3)
    model = {
        'training_error': training_error,
        'data_loss': total_loss,
        'saver': saver,
        'summary': merged,
        'our_image': our_image,
        'is_flip_lr': is_flip_lr,
        'is_flip_ud': is_flip_ud,
        'avg_op': avg_op,
        'slice_avg': slice_avg,
    }

    return model

def mask_loss(x, labels, masks):
    cnt_nonzero = tf.to_float(tf.count_nonzero(masks))
    loss = tf.reduce_sum(tf.multiply(tf.math.pow(x - labels, 2), masks)) / cnt_nonzero
    return loss

def data_arg(x, is_flip_lr, is_flip_ud):
    x = tf.cond(is_flip_lr > 0, lambda: tf.image.flip_left_right(x), lambda: x)
    x = tf.cond(is_flip_ud > 0, lambda: tf.image.flip_up_down(x), lambda: x)
    return x
