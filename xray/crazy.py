import torchxrayvision as xrv
import skimage.io
import torch
import torchvision
import numpy as np

# ----------- Step 1: Load and Preprocess the Image -----------
# Load image (e.g., a JPG X-ray image)
img_path = "16747_3_1.jpg"
img = skimage.io.imread(img_path)

# Normalize to match model's expected input
img = xrv.datasets.normalize(img, 255)  # Convert from [0, 255] to model's expected scale

# Convert to single-channel grayscale if RGB
if len(img.shape) == 3 and img.shape[2] == 3:
    img = img.mean(2)  # Convert to grayscale

img = img[None, ...]  # Add channel dimension (1, H, W)

# Resize and center-crop the image to 224x224
transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])
img = transform(img)

# Convert to PyTorch tensor and add batch dimension
img_tensor = torch.from_numpy(img).unsqueeze(0)  # Shape: (1, 1, 224, 224)

# ----------- Step 2: Load Pretrained Model -----------
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()  # Set model to evaluation mode

# ----------- Step 3: Run Inference -----------
with torch.no_grad():
    outputs = model(img_tensor)

# ----------- Step 4: Interpret Results -----------
threshold = 0.5  # Set a threshold for prediction confidence
results = dict(zip(model.pathologies, outputs[0].numpy()))

# Filter for likely positive predictions
positive_findings = {disease: float(score) for disease, score in results.items() if score >= threshold}

# Print results
if positive_findings:
    print("Predicted positive conditions (confidence >= 0.5):")
    for disease, score in positive_findings.items():
        print(f"- {disease}: {score:.2f}")
else:
    print("No positive findings detected above threshold.")
