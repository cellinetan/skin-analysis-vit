import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
import os

# CONFIG
DATASET_PATH = "patch_dataset"
BEST_MODEL_PATH = "vit_skin_model_best.pth"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.00005

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

# DATASET
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
print("Classes:", dataset.classes)
print("Total images:", len(dataset))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# MODEL
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.head = nn.Linear(model.head.in_features, len(dataset.classes))
model = model.to(device)

# Load best model sebelumnya
if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print(f"Loaded previous best model from: {BEST_MODEL_PATH}")
else:
    print("Previous best model not found. Training will start from pretrained ViT.")

# LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# TRAINING LOOP
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # VALIDASI
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Val Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved with acc: {best_acc:.2f}%")

print("Training selesai.")