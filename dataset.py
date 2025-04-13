# dataset.py
import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class BatikDatasetWithMixUp(Dataset):
    def __init__(self, image_dir, mixup_dir, mixup_csv, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load original + augmented (non-mixup) data
        self.samples = []
        self.labels = []
        class_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(image_dir))) if os.path.isdir(os.path.join(image_dir, cls))}
        self.class_to_idx = class_to_idx

        for cls in class_to_idx:
            cls_path = os.path.join(image_dir, cls)
            for fname in os.listdir(cls_path):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    self.samples.append(os.path.join(cls_path, fname))
                    label = torch.zeros(len(class_to_idx))
                    label[class_to_idx[cls]] = 1.0
                    self.labels.append(label)

        # Load mixup data
        if mixup_csv and mixup_dir:
            df = pd.read_csv(mixup_csv)
            for _, row in df.iterrows():
                img_path = os.path.join(mixup_dir, row['filename'])
                label = torch.tensor(row[1:].values.astype(float))  # one row of soft labels
                self.samples.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
