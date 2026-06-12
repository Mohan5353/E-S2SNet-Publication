import os
import argparse
import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import network.Punet
import numpy as np
import util
import cv2

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SAVE = 100
N_STEP = 1000

def train(file_path, save_dir, dropout_rate=0.3, sigma=-1, is_realnoisy=True, alpha=0.3, beta=0.05, n_step=1000, mask_prob=0.5, reg_layer='dec_conv2b', n_prediction=100):
    print(f"Self2Self on {file_path}")
    tf.reset_default_graph()
    gt = util.load_np_image(file_path)
    noisy = gt # since these are real noisy images (OCT, NR206), sigma=-1 uses input as noisy directly
    
    file_name = os.path.basename(file_path)
    model_path = os.path.join(save_dir, file_name.split('.')[0] + "/")
    os.makedirs(model_path, exist_ok=True)
    
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy, alpha, beta, mask_prob, reg_layer)
    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter(os.path.join(model_path, 'logs'), sess.graph)
        sess.run(tf.global_variables_initializer())
        for step in range(n_step):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            writer.add_summary(merged, step)
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:
                print(f"Step {step+1}, Loss {avg_loss/N_SAVE}")
                avg_loss = 0
                
        # Final evaluation (Inference Passes)
        sum_img = np.float32(np.zeros(our_image.shape.as_list()))
        for j in range(n_prediction):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            o_avg, o_image, merged_inf = sess.run([slice_avg, our_image, summay], feed_dict=feet_dict)
            writer.add_summary(merged_inf, n_step + j) # Continue logging past training steps
            sum_img += o_image
        # Apply min-max contrast stretching to avoid washed out images
        o_avg_sq = np.squeeze(o_avg)
        o_min = np.min(o_avg_sq)
        o_max = np.max(o_avg_sq)
        if o_max > o_min:
            o_avg_norm = (o_avg_sq - o_min) / (o_max - o_min)
        else:
            o_avg_norm = o_avg_sq
        o_avg = np.uint8(np.clip(o_avg_norm, 0, 1) * 255)
        cv2.imwrite(os.path.join(save_dir, f'S2S_{file_name}'), o_avg)
        saver.save(sess, os.path.join(model_path, "model.ckpt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--n_step', type=int, default=1000)
    parser.add_argument('--mask_prob', type=float, default=0.5)
    parser.add_argument('--reg_layer', type=str, default='dec_conv2b')
    parser.add_argument('--n_prediction', type=int, default=100)
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.data_dir, '*'))
    for f in files:
        if os.path.isfile(f):
            train(f, args.save_dir, dropout_rate=args.dropout_rate, sigma=-1, is_realnoisy=True, alpha=args.alpha, beta=args.beta, n_step=args.n_step, mask_prob=args.mask_prob, reg_layer=args.reg_layer, n_prediction=args.n_prediction)
