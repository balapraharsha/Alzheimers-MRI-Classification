import os
import shutil
import random

# Config
raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
preprocessed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed")

raw_dir = os.path.abspath(raw_dir)
preprocessed_dir = os.path.abspath(preprocessed_dir)

print(f"[INFO] Raw folder path: {raw_dir}")
print(f"[INFO] Preprocessed folder path: {preprocessed_dir}")

split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}
image_extensions = ('.jpg', '.jpeg', '.png')

# Get class folders
classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
print(f"[INFO] Detected classes: {classes}")

# create processed folders
for folder in ['train', 'val', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(preprocessed_dir, folder, cls), exist_ok=True)

# Split images
for cls in classes:
    class_dir = os.path.join(raw_dir, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(image_extensions)]
    
    if len(images) == 0:
        print(f"[WARNING] No images found in {class_dir}. Skipping class '{cls}'.")
        continue
    
    print(f"[INFO] Processing '{cls}' with {len(images)} images.")
    
    random.shuffle(images)
    n_total = len(images)
    n_train = int(n_total * split_ratio["train"])
    n_val = int(n_total * split_ratio["val"])
    n_test = n_total - n_train - n_val
    
    # Split images
    train_images = images[:n_train]
    val_images = images[n_train:n_train+n_val]
    test_images = images[n_train+n_val:]
    
    # Copy images
    for img in train_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(preprocessed_dir, "train", cls, img))
    for img in val_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(preprocessed_dir, "val", cls, img))
    for img in test_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(preprocessed_dir, "test", cls, img))
    
    print(f"[INFO] '{cls}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images copied.")

print("\n[INFO] Dataset successfully split into train, val, test!")
