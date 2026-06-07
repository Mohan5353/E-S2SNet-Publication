import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import network.Punet

tf.reset_default_graph()
noisy = np.random.rand(1, 448, 896, 1).astype(np.float32)
model = network.Punet.build_denoising_unet(noisy, 0.5, True)

print("E-S2SNet network built successfully!")
