import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm

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

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

base_dataset = datasets.ImageFolder(DATASET_PATH)
train_dataset_full = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
val_dataset_full = datasets.ImageFolder(DATASET_PATH, transform=val_transform)

class_names = base_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Total images:", len(base_dataset))

if num_classes != 3:
    raise ValueError(
        f"Dataset harus punya 3 kelas: normal, moderate, severe. "
        f"Sekarang terbaca {num_classes} kelas: {class_names}"
    )

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

targets = np.array(base_dataset.targets)

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

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

train_targets = targets[train_indices]
train_counts = Counter(train_targets)

class_weights = []
for i in range(num_classes):
    class_count = train_counts[i]
    weight = len(train_targets) / (num_classes * class_count)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print("\nTrain count:")
for class_index, class_name in enumerate(class_names):
    print(f"{class_name}: {train_counts[class_index]}")

print("\nClass weights:")
for class_name, weight in zip(class_names, class_weights):
    print(f"{class_name}: {weight.item():.4f}")

model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.head = nn.Linear(model.head.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

best_val_acc = 0.0

history = {
    "train_loss": [],
    "val_loss": [],
    "val_acc": []
}


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

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item()
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
                confusion_matrix[true_label.long(), pred_label.long()] += 1

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    acc = 100 * correct / total if total > 0 else 0

    return avg_loss, acc, confusion_matrix


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

    avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
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

print("\nPer-class validation result:")

for i, class_name in enumerate(class_names):
    tp = confusion_matrix[i, i].item()
    total_actual = confusion_matrix[i, :].sum().item()
    total_predicted = confusion_matrix[:, i].sum().item()

    recall = tp / total_actual if total_actual > 0 else 0
    precision = tp / total_predicted if total_predicted > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    print(
        f"{class_name} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1-score: {f1:.4f}"
    )

with open("training_history.json", "w") as f:
    json.dump(history, f, indent=4)

print("\nSaved:")
print(f"- {BEST_MODEL_PATH}")
print(f"- {CLASS_NAMES_PATH}")
print("- training_history.json")