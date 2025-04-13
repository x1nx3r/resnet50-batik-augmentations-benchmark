import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import pandas as pd

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# Parameters
test_dir = "./test_data"
batch_size = 32
model_path = "modelB_best.pth"
output_csv = "test_predictionsB.csv"

# Label names (update if needed based on your dataset class folders)
class_names = sorted(os.listdir(test_dir))  # assumes test_data has subfolders per class

# Transforms (same as training: resize + tensor)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset and loader
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 6)  # 6 Batik classes
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Predict all test images
results = []

sample_index = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        for i in range(imgs.size(0)):
            img_path, _ = test_dataset.samples[sample_index]
            img_filename = os.path.basename(img_path)
            true_class = class_names[labels[i].item()]
            pred_class = class_names[preds[i].item()]
            results.append({
                "filename": img_filename,
                "true_label": true_class,
                "predicted_label": pred_class
            })
            sample_index += 1

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")

