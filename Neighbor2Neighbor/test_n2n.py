import os
import glob
import cv2
import numpy as np
import torch
from torchvision import transforms
from arch_unet import UNet
import argparse

def test(model_path, data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    network = UNet(in_nc=3, out_nc=3, n_feature=48).cuda()
    network.load_state_dict(torch.load(model_path))
    network.eval()
    
    files = glob.glob(os.path.join(data_dir, '*.*'))
    for f in files:
        im = cv2.imread(f)
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_tensor = transforms.ToTensor()(im_rgb).unsqueeze(0).cuda()
        with torch.no_grad():
            pred = network(im_tensor)
        pred = pred.squeeze().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
        pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    test(args.model_path, args.data_dir, args.out_dir)
