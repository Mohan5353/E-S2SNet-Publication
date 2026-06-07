import os
import cv2
import glob

def process_dir(src_dir, dst_dir, target_shape=(896, 448)):
    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob(os.path.join(src_dir, '*.*'))
    for f in files:
        img = cv2.imread(f)
        if img is None: continue
        # Resize to target shape (W, H)
        img_resized = cv2.resize(img, target_shape)
        fname = os.path.basename(f)
        # Save as png for easier loading in some scripts
        fname = fname.split('.')[0] + '.png'
        cv2.imwrite(os.path.join(dst_dir, fname), img_resized)

if __name__ == '__main__':
    process_dir('./SD-OCT/raw', './SD-OCT/raw_fullres', target_shape=(896, 448))
    # Also prepare the averaged (ground truth) if needed. Wait, SD-OCT has a 'average' or 'ground_truth' directory?
    # Let's check what's inside SD-OCT.
    process_dir('./SD-OCT/average', './SD-OCT/average_fullres', target_shape=(896, 448))
