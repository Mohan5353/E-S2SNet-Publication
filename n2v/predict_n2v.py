import os
import argparse
from n2v.models import N2V
import cv2
import numpy as np
import glob

def predict(data_dir, save_path, log_name):
    # Load model
    model = N2V(config=None, name=log_name, basedir=save_path)
    
    out_dir = os.path.join(save_path, log_name, 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(data_dir, '*.*')))
    for f in files:
        img = cv2.imread(f)
        if img is None: continue
        pred = model.predict(img, axes='YXC')
        pred = np.clip(pred, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), pred)
    print(f"Generated predictions for N2V in {out_dir}")

if __name__ == '__main__':
    predict("../NR206/train_256", "results", "n2v_nr206")
