from collections import defaultdict

counts = defaultdict(int)

for class_name in class_names:
    class_dir = Path("dataset") / class_name
    print(f"🔍 Scanning {class_dir}...")
    image_paths = list(class_dir.rglob("*"))
    for p in image_paths:
        print(f" - Found: {p.name}, Suffix: {p.suffix}, Is file: {p.is_file()}")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            all_images.append((str(p), class_to_idx[class_name]))
            counts[class_name] += 1

for k, v in counts.items():
    print(f"{k}: {v} valid images")

print(f"Total images found: {len(all_images)}")
