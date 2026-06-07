import sys
import os
import cv2
import numpy as np
from PIL import Image
import re

def hfen_l1(pred, target):
    pred = np.float32(pred)
    target = np.float32(target)
    gaussian_target = cv2.GaussianBlur(target, (15, 15), 1.5)
    log_target = cv2.Laplacian(gaussian_target, cv2.CV_32F)
    gaussian_pred = cv2.GaussianBlur(pred, (15, 15), 1.5)
    log_pred = cv2.Laplacian(gaussian_pred, cv2.CV_32F)
    norm = np.sum(np.abs(log_target))
    if norm == 0:
        return 0.0
    return float(np.sum(np.abs(log_target - log_pred)) / norm)


# We will extract the logic from msr_cnr.py
# First, extract boxes
boxes = []
import os
with open(os.path.join(os.path.dirname(__file__), 'msr_cnr.py'), 'r') as f:
    content = f.read()
    
# Split by im=Image.open
sections = content.split('im=Image.open(image')[1:]
for s in sections:
    b1 = re.search(r'box1=\((.*?)\)', s)
    b2 = re.search(r'box2=\((.*?)\)', s)
    b3 = re.search(r'box3=\((.*?)\)', s)
    b4 = re.search(r'box4=\((.*?)\)', s)
    
    if b1 and b2 and b3 and b4:
        boxes.append((
            tuple(map(int, b1.group(1).split(','))),
            tuple(map(int, b2.group(1).split(','))),
            tuple(map(int, b3.group(1).split(','))),
            tuple(map(int, b4.group(1).split(','))),
        ))

print(f"Parsed {len(boxes)} box sets from msr_cnr.py")

# Now define the metric functions
def std_mean_back(a): 
    a = np.asanyarray(a) 
    m = np.mean(a) 
    sd = np.std(a) 
    return m, sd

def std_mean_fore(a, b, c): 
    a = np.asanyarray(a) 
    b = np.asanyarray(b) 
    c = np.asanyarray(c) 
    d = np.append(a, b)
    d = np.append(d, c)
    m = np.mean(d) 
    sd = np.std(d) 
    return m, sd

def corr(a, b):
    return np.sum(np.multiply(a, b))

def msr_cnr(im, box1, box2, box3, box4):
    im_back=np.array(im.crop(box1))[:,:,0] if np.array(im.crop(box1)).ndim == 3 else np.array(im.crop(box1))
    im_fore1=np.array(im.crop(box2))[:,:,0] if np.array(im.crop(box2)).ndim == 3 else np.array(im.crop(box2))
    im_fore2=np.array(im.crop(box3))[:,:,0] if np.array(im.crop(box3)).ndim == 3 else np.array(im.crop(box3))
    im_fore3=np.array(im.crop(box4))[:,:,0] if np.array(im.crop(box4)).ndim == 3 else np.array(im.crop(box4))

    mean_back, std_back = std_mean_back(im_back)
    mean_fore, std_fore = std_mean_fore(im_fore1, im_fore2, im_fore3)
    msr = (mean_fore/std_fore) if std_fore != 0 else 0
    cnr = (mean_fore-mean_back) / np.sqrt(0.5 * ((std_back*std_back) + (std_fore*std_fore)))
    return msr, cnr

def texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4):
    ddepth = cv2.CV_16S
    kernel_size = 3
    
    a=im.crop(box1)
    b=im.crop(box2)
    c=im.crop(box3)
    d=im.crop(box4)
    
    a1=input_im.crop(box1)
    b1=input_im.crop(box2)
    c1=input_im.crop(box3)
    d1=input_im.crop(box4)

    m1, std1 = std_mean_back(np.array(a)[:,:,0] if np.array(a).ndim==3 else np.array(a))
    m2, std2 = std_mean_back(np.array(b)[:,:,0] if np.array(b).ndim==3 else np.array(b))
    m3, std3 = std_mean_back(np.array(c)[:,:,0] if np.array(c).ndim==3 else np.array(c))
    m4, std4 = std_mean_back(np.array(d)[:,:,0] if np.array(d).ndim==3 else np.array(d))
    
    m11, std11 = std_mean_back(np.array(a1)[:,:,0] if np.array(a1).ndim==3 else np.array(a1))
    m12, std12 = std_mean_back(np.array(b1)[:,:,0] if np.array(b1).ndim==3 else np.array(b1))
    m13, std13 = std_mean_back(np.array(c1)[:,:,0] if np.array(c1).ndim==3 else np.array(c1))
    m14, std14 = std_mean_back(np.array(d1)[:,:,0] if np.array(d1).ndim==3 else np.array(d1))

    m_im, std_im = std_mean_back(np.array(im)[:,:,0] if np.array(im).ndim==3 else np.array(im)) 
    m_input_im, std_input_im = std_mean_back(np.array(input_im)[:,:,0] if np.array(input_im).ndim==3 else np.array(input_im)) 

    tp1 = (std1/std11) * np.sqrt((m_im/m_input_im)) if std11!=0 and m_input_im!=0 else 0
    tp2 = (std2/std12) * np.sqrt((m_im/m_input_im)) if std12!=0 and m_input_im!=0 else 0
    tp3 = (std3/std13) * np.sqrt((m_im/m_input_im)) if std13!=0 and m_input_im!=0 else 0
    tp4 = (std4/std14) * np.sqrt((m_im/m_input_im)) if std14!=0 and m_input_im!=0 else 0

    l1 = cv2.Laplacian(np.asarray(a.convert('L')), ddepth, ksize=kernel_size)
    l2 = cv2.Laplacian(np.asarray(b.convert('L')), ddepth, ksize=kernel_size)
    l3 = cv2.Laplacian(np.asarray(c.convert('L')), ddepth, ksize=kernel_size)
    l4 = cv2.Laplacian(np.asarray(d.convert('L')), ddepth, ksize=kernel_size)

    l1 = l1 - np.mean(l1)
    l2 = l2 - np.mean(l2)
    l3 = l3 - np.mean(l3)
    l4 = l4 - np.mean(l4)

    l11 = cv2.Laplacian(np.asarray(a1.convert('L')), ddepth, ksize=kernel_size)
    l21 = cv2.Laplacian(np.asarray(b1.convert('L')), ddepth, ksize=kernel_size)
    l31 = cv2.Laplacian(np.asarray(c1.convert('L')), ddepth, ksize=kernel_size)
    l41 = cv2.Laplacian(np.asarray(d1.convert('L')), ddepth, ksize=kernel_size)

    l11 = l11 - np.mean(l11)
    l21 = l21 - np.mean(l21)
    l31 = l31 - np.mean(l31)
    l41 = l41 - np.mean(l41)
    
    ep1 = corr(l11, l1)/np.sqrt(corr(l11, l11) * corr(l1, l1)) if (corr(l11, l11) * corr(l1, l1))>0 else 0
    ep2 = corr(l21, l2)/np.sqrt(corr(l21, l21) * corr(l2, l2)) if (corr(l21, l21) * corr(l2, l2))>0 else 0
    ep3 = corr(l31, l3)/np.sqrt(corr(l31, l31) * corr(l3, l3)) if (corr(l31, l31) * corr(l3, l3))>0 else 0
    ep4 = corr(l41, l4)/np.sqrt(corr(l41, l41) * corr(l4, l4)) if (corr(l41, l41) * corr(l4, l4))>0 else 0

    return (tp2+tp3+tp4)/3, (m1 * m1)/(std1 * std1), (ep2+ep3+ep4)/3

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_method(method_name, pred_dir, raw_dir, truth_dir):
    msr_list, cnr_list, tp_list, ep_list, ssim_list, psnr_list, hfen_list = [], [], [], [], [], [], []
    for i in range(1, 19):
        # image shapes might differ (png vs tif)
        # However, prepare_data_fullres.py saved as 1.png
        if method_name == 'E-S2SNet':
            im_path = os.path.join(pred_dir, f'S2S_{i}.png')
            if not os.path.exists(im_path): im_path = os.path.join(pred_dir, f'S2S_{i}.tif')
        else:
            im_path = os.path.join(pred_dir, f'{i}.png')
            if not os.path.exists(im_path): im_path = os.path.join(pred_dir, f'{i}.tif')
        
        raw_path = os.path.join(raw_dir, f'{i}.tif')
        if not os.path.exists(raw_path): raw_path = os.path.join(raw_dir, f'{i}.png')
        truth_path = os.path.join(truth_dir, f'{i}.tif')
        if not os.path.exists(truth_path): truth_path = os.path.join(truth_dir, f'{i}.png')
        
        if not os.path.exists(im_path) or not os.path.exists(raw_path):
            continue
            
        im = Image.open(im_path).convert("L")
        input_im = Image.open(raw_path).convert("L")
        
        # Calculate MSR, CNR, TP, EP
        # Scale boxes from 900x450 to 256x256
        def scale_box(b):
            x1, y1, x2, y2 = b
            return (int(x1 * 256 / 900), int(y1 * 256 / 450), int(x2 * 256 / 900), int(y2 * 256 / 450))
            
        box1, box2, box3, box4 = boxes[i-1]
        box1, box2, box3, box4 = scale_box(box1), scale_box(box2), scale_box(box3), scale_box(box4)
        
        m, c = msr_cnr(im, box1, box2, box3, box4)
        t, enl, ep = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
        
        msr_list.append(m)
        cnr_list.append(c)
        tp_list.append(t)
        ep_list.append(ep)
        
        if os.path.exists(truth_path):
            t_im = Image.open(truth_path).convert("L")
            pred_im = im.convert("L")
            t_arr = np.array(t_im)
            p_arr = np.array(pred_im)
            psnr_list.append(peak_signal_noise_ratio(t_arr, p_arr))
            ssim_list.append(structural_similarity(t_arr, p_arr))
            hfen_list.append(hfen_l1(p_arr, t_arr))
            
    print(f"--- {method_name} ---")
    print(f"MSR:  {np.mean(msr_list):.4f}")
    print(f"CNR:  {np.mean(cnr_list):.4f}")
    print(f"TP:   {np.mean(tp_list):.4f}")
    print(f"EP:   {np.mean(ep_list):.4f}")
    if len(psnr_list) > 0:
        print(f"PSNR: {np.mean(psnr_list):.4f}")
        print(f"SSIM: {np.mean(ssim_list):.4f}")
    print("")
    
    return {
        "MSR": float(np.mean(msr_list)),
        "CNR": float(np.mean(cnr_list)),
        "TP": float(np.mean(tp_list)),
        "EP": float(np.mean(ep_list)),
        "PSNR": float(np.mean(psnr_list)) if len(psnr_list) > 0 else 0.0,
        "SSIM": float(np.mean(ssim_list)) if len(ssim_list) > 0 else 0.0,
        "HFEN": float(np.mean(hfen_list)) if len(hfen_list) > 0 else 0.0
    }

if __name__ == '__main__':
    evaluate_method('Neighbor2Neighbor', 'Neighbor2Neighbor/results/predictions_sdoct', 'SD-OCT/raw_cropped', 'SD-OCT/average_cropped')
    evaluate_method('Blind2Unblind', 'Blind2Unblind/results/predictions_sdoct', 'SD-OCT/raw_cropped', 'SD-OCT/average_cropped')
    evaluate_method('noise2self', 'noise2self/results_sdoct_fullres/predictions', 'SD-OCT/raw_cropped', 'SD-OCT/average_cropped')
    evaluate_method('n2v', 'n2v/results_sdoct_fullres/n2v_sdoct_fullres/predictions', 'SD-OCT/raw_cropped', 'SD-OCT/average_cropped')
    evaluate_method('E-S2SNet', 'Self2Self/results_sdoct_fullres', 'SD-OCT/raw_cropped', 'SD-OCT/average_cropped')
