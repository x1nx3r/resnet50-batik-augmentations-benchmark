import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# Parameters
resize_dim = (500, 500)
crop_size = 224
rotation_range = 30
output_dir = "processed_dataset"
os.makedirs(output_dir, exist_ok=True)

# Define class names
class_names = sorted([d.name for d in Path("dataset_split/train").iterdir() if d.is_dir()])
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
all_images = []

# Load all image paths and their labels
allowed_exts = {".jpg", ".jpeg", ".png"}

for class_name in class_names:
    class_dir = Path("dataset_split/train") / class_name
    image_paths = [p for p in class_dir.rglob("*") if p.suffix.lower() in allowed_exts]
    
    print(f"{class_name}: {len(image_paths)} images")
    for path in image_paths:
        all_images.append((str(path), class_to_idx[class_name]))


print(f"Total images found: {len(all_images)}")

# Helper functions
def random_crop(img, size):
    h, w = img.shape[:2]
    if h < size or w < size:
        raise ValueError(f"Image too small for cropping: {h}x{w} < {size}")
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)
    return img[top:top+size, left:left+size]

def random_rotate(img, angle_range):
    angle = random.uniform(-angle_range, angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# Apply augmentation
for img_path, label_idx in tqdm(all_images):
    class_name = class_names[label_idx]
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Skipping unreadable image: {img_path}")
        continue
    img = cv2.resize(img, resize_dim)

    base_filename = Path(img_path).stem
    out_dir = os.path.join(output_dir, class_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save resized original
    cv2.imwrite(os.path.join(out_dir, f"{base_filename}_orig.png"), img)

    # Random Crop
    try:
        cropped = random_crop(img, crop_size)
        cv2.imwrite(os.path.join(out_dir, f"{base_filename}_crop.png"), cropped)
    except ValueError as e:
        print(f"⚠️ {e}")

    # Random Rotate
    rotated = random_rotate(img, rotation_range)
    cv2.imwrite(os.path.join(out_dir, f"{base_filename}_rotate.png"), rotated)

print("✅ Preprocessing complete!")
