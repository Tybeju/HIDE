import glob
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
import os

class HIDEDataset(Dataset):
    def __init__(self, data_type, img_size, limit=None):
        assert data_type in ("train", "test")
        self.data_type = data_type
        self.img_size = img_size
        
        self.blurred_images = []
        self.sharp_images = []
        
        data_dir = f"HIDE_dataset/{data_type}/"
        gt_dir = "HIDE_dataset/GT/"
        
        if data_type == "train":
            image_list = glob.glob(data_dir + "*.png")
            if limit is not None: 
                image_list = image_list[:limit]
            for img_path in image_list:
                sharp_path = gt_dir + os.path.basename(img_path)
                self.blurred_images.append(img_path)
                self.sharp_images.append(sharp_path)

    def __len__(self):
        return len(self.sharp_images)

    def __getitem__(self, idx):
        blurred_image_path = self.blurred_images[idx]
        sharp_image_path = self.sharp_images[idx]
        
        blurred_img = cv2.imread(blurred_image_path)
        sharp_img = cv2.imread(sharp_image_path)
        
        blurred_img = cv2.resize(blurred_img, self.img_size)
        sharp_img = cv2.resize(sharp_img, self.img_size)
        
        blurred_img_tensor = torch.from_numpy(np.transpose(blurred_img, (2, 0, 1))).float() / 255.0
        sharp_img_tensor = torch.from_numpy(np.transpose(sharp_img, (2, 0, 1))).float() / 255.0
        
        return blurred_img_tensor, sharp_img_tensor

