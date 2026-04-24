from PIL import Image
import os

folders = [
    "patch_dataset/normal",
    "patch_dataset/acne",
    "patch_dataset/oily",
    "patch_dataset/dry"
]

for folder in folders:
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)

            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224))

            img.save(path)

print("done")