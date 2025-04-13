import os
import shutil
import random
from pathlib import Path

# CONFIGURATIONS
SOURCE_DIR = "raw_dataset"            # Folder asli yang berisi 6 subfolder kelas
OUTPUT_DIR = "dataset_split"          # Folder keluaran
TRAIN_RATIO = 0.8                     # 80% train, 20% test
SEED = 42

random.seed(SEED)

# Buat folder keluaran
train_dir = os.path.join(OUTPUT_DIR, "train")
test_dir = os.path.join(OUTPUT_DIR, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Mulai proses split
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    n_train = int(len(images) * TRAIN_RATIO)
    train_images = images[:n_train]
    test_images = images[n_train:]

    # Buat folder kelas di output
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Salin file
    for img_name in train_images:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(train_dir, class_name, img_name)
        shutil.copy2(src, dst)

    for img_name in test_images:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(test_dir, class_name, img_name)
        shutil.copy2(src, dst)

    print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

print("✅ Dataset split selesai.")
