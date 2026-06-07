import os
import cv2
import numpy as np

def process_image(img):
    # Resize keeping aspect ratio
    h, w = img.shape[:2]
    if h < w:
        new_h = 256
        new_w = int(w * (256 / h))
    else:
        new_w = 256
        new_h = int(h * (256 / w))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center crop 256x256
    y1 = (new_h - 256) // 2
    y2 = y1 + 256
    x1 = (new_w - 256) // 2
    x2 = x1 + 256
    img_cropped = img_resized[y1:y2, x1:x2]
    return img_cropped

def process_dir(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if os.path.isfile(p):
            img = cv2.imread(p)
            if img is not None:
                img_proc = process_image(img)
                cv2.imwrite(os.path.join(dst, f), img_proc)
            else:
                print(f"Failed to read {p}")

if __name__ == "__main__":
    process_dir('/media/mmk/DATA-1/E-S2SNet/SD-OCT/raw', '/media/mmk/DATA-1/E-S2SNet/SD-OCT/raw_256')
    process_dir('/media/mmk/DATA-1/E-S2SNet/NR206/train', '/media/mmk/DATA-1/E-S2SNet/NR206/train_256')
    process_dir('/media/mmk/DATA-1/E-S2SNet/NR206/val', '/media/mmk/DATA-1/E-S2SNet/NR206/val_256')
    process_dir('/media/mmk/DATA-1/E-S2SNet/NR206/test', '/media/mmk/DATA-1/E-S2SNet/NR206/test_256')
    print("Done preparing datasets.")
