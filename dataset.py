from typing import List, Optional, Tuple
import glob
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
import os

# TODO: add augment. (?) - gauss., rot., moire, ...


class HIDEDataset(Dataset):
    def __init__(
        self,
        data_type: str,
        img_size: Tuple[int, int],
        test_set: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        assert data_type in ("train", "test")
        assert test_set in ("short", "long", None)

        self.data_type = data_type
        self.img_size = img_size

        self.blurred_images: List[str] = []
        self.sharp_images: List[str] = []

        data_dir = f"HIDE_dataset/{data_type}/"
        gt_dir = "HIDE_dataset/GT/"
        test_subdirs = {"short": "/test-close-ups/", "long": "/test-long-shot/"}

        image_list = []

        if data_type == "train":
            image_list = self._load_images(data_dir + "*.png", limit)
        elif data_type == "test":
            if test_set is None:
                image_list = self._load_images(
                    data_dir + test_subdirs["short"] + "*.png", None
                ) + self._load_images(data_dir + test_subdirs["long"] + "*.png", None)
            else:
                image_list = self._load_images(
                    data_dir + test_subdirs[test_set] + "*.png", limit
                )

        for img_path in image_list:
            sharp_path = gt_dir + os.path.basename(img_path)
            self.blurred_images.append(img_path)
            self.sharp_images.append(sharp_path)

    def _load_images(self, path_pattern: str, limit: Optional[int]) -> List[str]:
        image_list = glob.glob(path_pattern)
        if limit is not None:
            image_list = image_list[:limit]
        return image_list

    def __len__(self) -> int:
        return len(self.sharp_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        blurred_img, sharp_img = [
            cv2.imread(p) for p in [self.blurred_images[idx], self.sharp_images[idx]]
        ]
        blurred_img, sharp_img = [
            cv2.resize(img, self.img_size) for img in [blurred_img, sharp_img]
        ]

        blurred_img_tensor, sharp_img_tensor = [
            torch.from_numpy(np.transpose(img, (2, 0, 1))).float() / 255.0
            for img in [blurred_img, sharp_img]
        ]

        return blurred_img_tensor, sharp_img_tensor
