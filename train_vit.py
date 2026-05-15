import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm

# CONFIG
DATASET_PATH = "patch_dataset"
BEST_MODEL_PATH = "vit_skin_model_best.pth"
CLASS_NAMES_PATH = "class_names.json"
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.00005
VAL_RATIO = 0.2
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Augmentation and transformations for data
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize gambar menjadi 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontal dengan 50% probabilitas
    transforms.RandomRotation(15),  # Rotasi acak antara -15° dan 15°
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
    ),
    transforms.ToTensor(),  # Convert ke tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize gambar menjadi 224x224
    transforms.ToTensor(),  # Convert ke tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Function to properly collate data (convert to tensor)
def collate_fn(batch):
    images, labels = zip(*batch)

    # Convert images list to tensor (ensure tensor format)
    images = torch.stack([transforms.ToTensor()(image) for image in images], dim=0)
    labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor

    return images, labels

# Load dataset
dataset = datasets.ImageFolder(DATASET_PATH)
class_names = dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Total images:", len(dataset))

if num_classes != 3:
    raise ValueError(f"Dataset harus punya 3 kelas: normal, moderate, severe. "
                     f"Sekarang terbaca {num_classes} kelas: {class_names}")

# Save class names
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

# Split dataset into train and validation
targets = np.array(dataset.targets)

print("\nDataset count:")
for class_index, class_name in enumerate(class_names):
    count = int(np.sum(targets == class_index))
    print(f"{class_name}: {count}")

train_indices = []
val_indices = []

for class_index in range(num_classes):
    class_indices = np.where(targets == class_index)[0].tolist()
    random.shuffle(class_indices)

    val_count = max(1, int(len(class_indices) * VAL_RATIO))

    val_indices.extend(class_indices[:val_count])
    train_indices.extend(class_indices[val_count:])

random.shuffle(train_indices)
random.shuffle(val_indices)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

# Check image shape to ensure it's (3, 224, 224)
sample_images, sample_labels = next(iter(train_loader))
print("Sample batch image shape:", sample_images.shape)
print("Sample batch label shape:", sample_labels.shape)

# Define the model
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.head = nn.Linear(model.head.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

best_val_acc = 0.0

history = {
    "train_loss": [],
    "val_loss": [],
    "val_acc": [],
}

# Evaluation function
def evaluate_model():
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
                confusion_matrix[true_label.long(), pred_label.long()] += 1

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    acc = 100 * correct / total if total > 0 else 0

    return avg_loss, acc, confusion_matrix

# Training loop
print("\nTraining started...")

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    val_loss, val_acc, confusion_matrix = evaluate_model()

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved: {BEST_MODEL_PATH} | Val Acc: {best_val_acc:.2f}%")

print("\nTraining selesai.")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

print("\nFinal confusion matrix:")
print("Rows = actual class, Columns = predicted class")
print("Class order:", class_names)
print(confusion_matrix.numpy())

with open("training_history.json", "w") as f:
    json.dump(history, f, indent=4)

print("\nSaved:")
print(f"- {BEST_MODEL_PATH}")
print(f"- {CLASS_NAMES_PATH}")
print("- training_history.json")