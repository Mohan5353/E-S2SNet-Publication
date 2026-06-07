import os
import glob
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mask import Masker
from models.babyunet import BabyUnet
import argparse
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, '*')))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        if img is None:
            return torch.zeros((3, 448, 896)), self.files[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype('float32') / 255.0
        return torch.tensor(img), self.files[idx]

def train(data_dir, save_path):
    os.makedirs(save_path, exist_ok=True)
    dataset = ImageDataset(data_dir)
    if len(dataset) == 0:
        print(f"No images found in {data_dir}")
        return
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = BabyUnet(3, 3).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    masker = Masker()
    
    print(f"Training noise2self on {data_dir}")
    for epoch in range(100): # Train properly
        for i, (batch, _) in enumerate(loader):
            noisy_images = batch.cuda()
            net_input, mask = masker.mask(noisy_images, i)
            output = model(net_input)
            loss = loss_function(output*mask, noisy_images*mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss {loss.item()}")
    
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    print(f"Saved to {save_path}")
    
    # Generate predictions
    model.eval()
    out_dir = os.path.join(save_path, 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch, fpaths in test_loader:
            noisy_images = batch.cuda()
            output = model(noisy_images)
            pred = output.cpu().numpy()[0].transpose(1, 2, 0)
            pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(fpaths[0])), pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    train(args.data_dir, args.save_path)
