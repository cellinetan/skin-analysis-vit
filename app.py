import io

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
import timm

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["acne", "dry", "normal", "oily"]

MODEL_PATH = "vit_skin_model_best.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = timm.create_model("vit_base_patch16_224", pretrained=False)
model.head = nn.Linear(model.head.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

mp_face_mesh = mp.solutions.face_mesh


def predict_patch(image):
    if image is None or image.size == 0:
        return {
            "label": "unknown",
            "confidence": 0,
            "probabilities": {
                "acne": 0,
                "dry": 0,
                "normal": 0,
                "oily": 0
            }
        }

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)[0]

    probabilities = {
        classes[i]: round(probs[i].item() * 100, 2)
        for i in range(len(classes))
    }

    top_index = torch.argmax(probs).item()
    top_label = classes[top_index]
    confidence = round(probs[top_index].item() * 100, 2)

    return {
        "label": top_label,
        "confidence": confidence,
        "probabilities": probabilities
    }


def crop_regions(image):
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark

    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]

    x_min = max(min(xs), 0)
    x_max = min(max(xs), w)
    y_min = max(min(ys), 0)
    y_max = min(max(ys), h)

    face_w = x_max - x_min
    face_h = y_max - y_min

    forehead = image[
        y_min + int(face_h * 0.05): y_min + int(face_h * 0.28),
        x_min + int(face_w * 0.28): x_min + int(face_w * 0.72)
    ]

    nose = image[
        y_min + int(face_h * 0.32): y_min + int(face_h * 0.62),
        x_min + int(face_w * 0.35): x_min + int(face_w * 0.65)
    ]

    left_cheek = image[
        y_min + int(face_h * 0.42): y_min + int(face_h * 0.72),
        x_min + int(face_w * 0.08): x_min + int(face_w * 0.38)
    ]

    right_cheek = image[
        y_min + int(face_h * 0.42): y_min + int(face_h * 0.72),
        x_min + int(face_w * 0.62): x_min + int(face_w * 0.92)
    ]

    return {
        "forehead": forehead,
        "nose": nose,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek
    }


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]
    image_bytes = file.read()

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({
            "success": False,
            "error": "Invalid image file"
        }), 400

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    regions = crop_regions(image)

    if regions is None:
        return jsonify({
            "success": False,
            "error": "No face detected. Please take a clear face photo."
        }), 400

    result = {
        "forehead": predict_patch(regions["forehead"]),
        "nose": predict_patch(regions["nose"]),
        "left_cheek": predict_patch(regions["left_cheek"]),
        "right_cheek": predict_patch(regions["right_cheek"])
    }

    return jsonify({
        "success": True,
        "message": "Prediction success",
        "result": result
    })


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Skin Analysis API is running",
        "model": "Vision Transformer",
        "classes": classes,
        "output": "label, confidence, probabilities"
    })


if __name__ == "__main__":
    print("Using device:", device)
    print("Loaded model:", MODEL_PATH)
    app.run(host="0.0.0.0", port=5000, debug=True)