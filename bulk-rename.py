import os

folder_path = "raw_dataset/batik-ceplok"
prefix = "batik_ceplok"  # ganti sesuai kelas
ext = ".jpg"             # ganti jika formatnya .png atau lainnya

# Ambil semua file gambar dan sort (optional)
images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(ext)])

# Rename loop
for idx, filename in enumerate(images, start=1):
    new_name = f"{prefix}_{idx:03d}{ext}"
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    os.rename(src, dst)
    print(f"Renamed: {filename} → {new_name}")
