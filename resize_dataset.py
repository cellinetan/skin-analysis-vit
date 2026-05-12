import os
import numpy as np
from PIL import Image, ImageOps

DATASET_DIR = "patch_dataset"
IMAGE_SIZE = 224

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def trim_white_border(image, threshold=245):
    image = image.convert("RGB")
    arr = np.array(image)

    mask = np.any(arr < threshold, axis=2)

    if not mask.any():
        return image

    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1

    cropped = image.crop((x_min, y_min, x_max, y_max))
    return cropped


def center_crop_square(image):
    image = image.convert("RGB")

    width, height = image.size
    min_side = min(width, height)

    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side

    return image.crop((left, top, right, bottom))


def process_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    image = trim_white_border(image)
    image = center_crop_square(image)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    return image


def resize_dataset_in_place():
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset folder not found: {DATASET_DIR}")
        return

    class_names = [
        folder for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
    ]

    if not class_names:
        print("No class folders found.")
        return

    print("Detected classes:", class_names)

    total_processed = 0
    total_skipped = 0

    for class_name in class_names:
        class_dir = os.path.join(DATASET_DIR, class_name)

        image_files = [
            file for file in os.listdir(class_dir)
            if file.lower().endswith(VALID_EXTENSIONS)
        ]

        print(f"\nProcessing class: {class_name}")
        print(f"Found images: {len(image_files)}")

        for index, file_name in enumerate(image_files, start=1):
            image_path = os.path.join(class_dir, file_name)

            try:
                with Image.open(image_path) as image:
                    processed_image = process_image(image)
                    processed_image.save(image_path, "JPEG", quality=95)

                total_processed += 1

                if index % 50 == 0:
                    print(f"Processed {index}/{len(image_files)} images")

            except Exception as e:
                total_skipped += 1
                print(f"Skipped: {image_path}")
                print(f"Reason: {e}")

    print("\nResize finished.")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")
    print(f"Images overwritten inside: {DATASET_DIR}")


if __name__ == "__main__":
    resize_dataset_in_place()