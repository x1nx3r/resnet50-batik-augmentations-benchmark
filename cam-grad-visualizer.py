import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
model_path = "modelA_best.pth"  # Path to your saved model
test_dir = "test_data"
output_dir = "heatmap_test_model_A"  # Directory where heatmaps will be saved

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Transform (should match training transforms!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load test dataset
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
class_names = test_dataset.classes

# Load the model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)  # Adjust this to your number of classes
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Function to generate Grad-CAM heatmap
def generate_cam(model, image, target_class):
    # Register hooks to capture the gradients and activations
    def hook_fn(module, input, output):
        activations.append(output)
        def backward_hook(grad):
            gradients.append(grad)

    activations = []
    gradients = []
    
    last_conv_layer = model.layer4[2].conv3  # Last convolutional layer
    hook = last_conv_layer.register_forward_hook(hook_fn)
    grad_hook = last_conv_layer.register_backward_hook(lambda module, grad_in, grad_out: gradients.append(grad_out[0]))

    # Forward pass
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    outputs = model(image)

    # Zero the gradients
    model.zero_grad()

    # Backward pass: calculate gradients for the target class
    target = outputs[0][target_class]
    target.backward()

    # Get the gradients and activations
    grad = gradients[0].cpu().data.numpy()[0]  # Gradient of shape (num_channels, height, width)
    activation = activations[0].cpu().data.numpy()[0]  # Activation of shape (num_channels, height, width)
    
    # Pool the gradients across all channels
    weights = np.mean(grad, axis=(1, 2))  # (num_channels, )

    # Create weighted sum of the activations
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activation[i, :, :]

    # Apply ReLU to the CAM
    cam = np.maximum(cam, 0)

    # Normalize the CAM between 0 and 1
    cam = cam / np.max(cam)

    # Resize CAM to match the image size
    cam = cv2.resize(cam, (224, 224))

    # Remove the hooks
    hook.remove()
    grad_hook.remove()

    return cam

# Visualize and save CAM for a sample image
def visualize_and_save_cam(image, target_class, index):
    # Generate the CAM
    cam = generate_cam(model, image, target_class)

    # Convert the image to numpy
    image = image.squeeze().cpu().numpy().transpose(1, 2, 0)

    # Plot original image and CAM side by side
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Original Image\nClass: {class_names[target_class]}")
    plt.axis('off')

    # CAM Image
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(cam, cmap='jet', alpha=0.5)  # Overlay the CAM
    plt.title(f"Class Activation Map\nClass: {class_names[target_class]}")
    plt.axis('off')

    # Save the resulting figure
    plt.tight_layout()
    heatmap_filename = os.path.join(output_dir, f"heatmap_{index}.png")
    plt.savefig(heatmap_filename)
    plt.close()

# Loop through the entire test dataset
index = 0
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    
    # Iterate through each image in the batch
    for i in range(len(images)):
        image = images[i]
        target_class = labels[i].item()
        
        # Visualize and save the CAM for this image
        visualize_and_save_cam(image, target_class, index)
        index += 1

print(f"All heatmaps saved in {output_dir} directory.")
