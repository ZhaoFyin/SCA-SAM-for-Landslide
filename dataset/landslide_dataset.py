import numpy as np
import random
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

CLASSES = ['no_landslide', 'landslide']


class LandslideDataset(Dataset):
    """
    - Inputs: VOC-like layout under {voc_root}/VOC_landslide
        JPEGImages/{id}.jpg
        SegmentationObject/{id}.png
        ImageSets/Segmentation/{train|val|test}.txt
    - Output:
        image: FloatTensor [3, 1024, 1024], range [0,1]
        mask : FloatTensor [1, 1024, 1024], values {0., 1.}
        size : (orig_h, orig_w) before resize (useful for SAM后续反变换)
    """
    def __init__(self, voc_root, txt_name: str = "train.txt"):
        self.root = os.path.join(voc_root, "VOC_landslide")
        txt_path = os.path.join(self.root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), f"file '{txt_path}' does not exist."

        with open(txt_path, "r") as f:
            self.file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.image_dir = os.path.join(self.root, "JPEGImages")
        self.mask_dir = os.path.join(self.root, "SegmentationObject")

        self.target_size = 512  # SAM 输入边长
        self.mode = txt_name.split(".")[0]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.image_dir, file_name + ".jpg")
        mask_path = os.path.join(self.mask_dir, file_name + ".png")

        # --- load ---
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 单通道

        # --- resize & pad ---
        w, h = image.size
        image_resized = TF.resize(image, size=[self.target_size, self.target_size], interpolation=InterpolationMode.BILINEAR)
        mask_resized = TF.resize(mask, size=[self.target_size, self.target_size], interpolation=InterpolationMode.NEAREST)

        # --- to tensor ---
        image_t = TF.to_tensor(image_resized).float()  # [0,1], [3,H,W]
        # 二值化（>0 视为前景）
        mask_np = np.array(mask_resized)
        mask_np[mask_np == 0] = 0
        mask_np[mask_np != 0] = 1

        mask_t = torch.from_numpy(mask_np).long()

        return {'img': image_t, 'gt_semantic_seg': mask_t, "size": (w, h), "img_id": file_name}
