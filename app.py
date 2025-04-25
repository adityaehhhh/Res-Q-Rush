from flask import Flask, request, jsonify
import torchxrayvision as xrv
import torchvision, torch, skimage
from PIL import Image
import numpy as np
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load model once
model = xrv.models.DenseNet(weights="densenet121-res224-all")
transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224)
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert("L")
    img = np.array(img)
    img = xrv.datasets.normalize(img, 255)
    img = img[None, ...]  # Add channel

    img = transform(img)
    img = torch.from_numpy(img).unsqueeze(0)  # Add batch

    with torch.no_grad():
        output = model(img)[0]

    preds = dict(zip(model.pathologies, output.numpy()))
    thresholded = {k: float(v) for k, v in preds.items() if v > 0.5}
    return jsonify({'results': thresholded})

if __name__ == '__main__':
    app.run(debug=True)
