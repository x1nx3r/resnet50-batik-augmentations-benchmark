import os
import cv2
import numpy as np
import random
from glob import glob
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Parameters
resize_dim = (500, 500)
crop_size = 224
rotation_range = 30
output_dir = "processed_dataset"
mixup_label_csv = os.path.join(output_dir, "mixup_labels.csv")
os.makedirs(output_dir, exist_ok=True)

# Define class names
class_names = sorted(os.listdir("dataset"))
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
all_images = []

# Load all image paths and their labels
for class_name in class_names:
    image_paths = glob(f"dataset/{class_name}/*.png") + glob(f"dataset/{class_name}/*.jpg") + glob(f"dataset/{class_name}/*.jpeg")
    for path in image_paths:
        all_images.append((path, class_to_idx[class_name]))

# Helper functions
def random_crop(img, size):
    h, w = img.shape[:2]
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)
    return img[top:top+size, left:left+size]

def random_rotate(img, angle_range):
    angle = random.uniform(-angle_range, angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def mixup_images(img1, img2, label1, label2):
    lam = np.random.beta(0.4, 0.4)
    mixed_img = (lam * img1 + (1 - lam) * img2).astype(np.uint8)
    soft_label = lam * label1 + (1 - lam) * label2
    return mixed_img, soft_label

# Create variations
mixup_records = []
for i, (img_path, label_idx) in enumerate(tqdm(all_images)):
    class_name = class_names[label_idx]
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize_dim)
    
    base_filename = Path(img_path).stem
    label_onehot = np.eye(len(class_names))[label_idx]

    # Save resized original
    out_dir = os.path.join(output_dir, class_name)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{base_filename}_orig.png"), img)

    # Random Crop
    cropped = random_crop(img, crop_size)
    cv2.imwrite(os.path.join(out_dir, f"{base_filename}_crop.png"), cropped)

    # Random Rotate
    rotated = random_rotate(img, rotation_range)
    cv2.imwrite(os.path.join(out_dir, f"{base_filename}_rotate.png"), rotated)

    # MixUp
    j = random.randint(0, len(all_images) - 1)
    while j == i:
        j = random.randint(0, len(all_images) - 1)
    mix_img_path, mix_label_idx = all_images[j]
    mix_img = cv2.imread(mix_img_path)
    mix_img = cv2.resize(mix_img, resize_dim)
    mix_label_onehot = np.eye(len(class_names))[mix_label_idx]

    mixed_img, soft_label = mixup_images(img, mix_img, label_onehot, mix_label_onehot)
    mixup_name = f"{base_filename}_MIX_{Path(mix_img_path).stem}.png"
    cv2.imwrite(os.path.join(output_dir, mixup_name), mixed_img)
    mixup_records.append([mixup_name] + soft_label.tolist())

# Save mixup labels
mixup_df = pd.DataFrame(mixup_records, columns=["filename"] + class_names)
mixup_df.to_csv(mixup_label_csv, index=False)

print("✅ Preprocessing complete!")
