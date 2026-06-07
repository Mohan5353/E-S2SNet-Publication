import os
import argparse
import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import network.Punet
import numpy as np
import util
import cv2

LEARNING_RATE = 1e-4

def train_custom():
    file_path = "../SD-OCT/raw_cropped/10.png"
    save_dir = "results_custom_img10"
    
    # Golden Mean Config
    dropout_rate = 0.3
    alpha = 0.1
    beta = 0.05
    gamma = 0.5
    mask_prob = 0.4
    reg_layer = 'dec_conv2b'
    
    n_step = 3000
    n_prediction = 500
    
    print(f"Starting custom training on {file_path}")
    tf.reset_default_graph()
    gt = util.load_np_image(file_path)
    noisy = gt
    
    file_name = os.path.basename(file_path)
    model_path = os.path.join(save_dir, "img10_model")
    os.makedirs(model_path, exist_ok=True)
    
    model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, True, alpha, beta, gamma, mask_prob, reg_layer)
    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(os.path.join(model_path, 'logs/train'), sess.graph)
        sess.run(tf.global_variables_initializer())
        
        avg_loss = 0
        for step in range(n_step):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            train_writer.add_summary(merged, step)
            avg_loss += loss_value
            
            if (step + 1) % 100 == 0:
                print(f"Training Step {step+1}, Loss {avg_loss/100:.4f}")
                avg_loss = 0
                
            # Perform inference at 1000, 2000, 3000
            if (step + 1) % 1000 == 0:
                print(f"--- Running 500 Inference Passes at Step {step+1} ---")
                inf_writer = tf.summary.FileWriter(os.path.join(model_path, f'logs/inference_at_{step+1}'), sess.graph)
                
                # Reset moving average for inference accumulation
                sess.run(slice_avg.assign(tf.zeros_like(slice_avg)))
                
                sum_img = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(n_prediction):
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                    # We run summary to capture energy loss during inference!
                    o_avg, o_image, inf_merged = sess.run([slice_avg, our_image, summay], feed_dict=feet_dict)
                    inf_writer.add_summary(inf_merged, j)
                    sum_img += o_image
                    
                o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))
                cv2.imwrite(os.path.join(save_dir, f'10_step_{step+1}.png'), o_avg)
                saver.save(sess, os.path.join(model_path, f"model_step_{step+1}.ckpt"))
                print(f"Saved inference output and checkpoint at Step {step+1}")

if __name__ == '__main__':
    train_custom()
