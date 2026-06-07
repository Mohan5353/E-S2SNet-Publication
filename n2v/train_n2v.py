import os
import argparse
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import cv2
import numpy as np

def train(data_dir, save_path, log_name):
    datagen = N2V_DataGenerator()
    imgs = datagen.load_imgs_from_directory(directory=data_dir, filter='*.*', dims='YXC')
    if len(imgs) == 0:
        print(f"No images found in {data_dir}")
        return
        
    for i in range(len(imgs)):
        imgs[i] = imgs[i][..., :3]
        
    X = datagen.generate_patches_from_list(imgs, shape=(64, 64))
    # use 10% for val
    X_val = X[:max(1, len(X)//10)]
    X = X[max(1, len(X)//10):]
        
    config = N2VConfig(X, unet_kern_size=3, 
                       train_steps_per_epoch=int(X.shape[0]/128),
                       train_epochs=100, train_loss='mse', batch_norm=True, 
                       train_batch_size=128, n2v_perc_pix=0.198, n2v_patch_shape=(64, 64), 
                       n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)
                       
    model = N2V(config=config, name=log_name, basedir=save_path)
    model.train(X, X_val)
    print(f"N2V trained on {data_dir}")
    
    # Save predicted full resolution images
    out_dir = os.path.join(save_path, log_name, 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, '*.*')))
    for f in files:
        img = cv2.imread(f)
        pred = model.predict(img, axes='YXC')
        # pred might be float, convert to uint8
        pred = np.clip(pred, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--log_name', type=str, required=True)
    args = parser.parse_args()
    train(args.data_dir, args.save_path, args.log_name)
