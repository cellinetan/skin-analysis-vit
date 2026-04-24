import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# class label (HARUS SAMA URUTAN)
classes = ['acne', 'dry', 'normal', 'oily']

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# load model
model = timm.create_model("vit_base_patch16_224", pretrained=False)
model.head = nn.Linear(model.head.in_features, len(classes))

model.load_state_dict(torch.load("vit_skin_model_best.pth", map_location=device))
model = model.to(device)
model.eval()

# load image
image = Image.open("test13.jpg").convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# predict
with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print("Prediction:", classes[predicted.item()])
print("Confidence:", round(confidence.item() * 100, 2), "%")