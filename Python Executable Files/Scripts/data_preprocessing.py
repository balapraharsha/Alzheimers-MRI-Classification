import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Config
preprocessed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed")
preprocessed_dir = os.path.abspath(preprocessed_dir)

img_height, img_width = 224, 224 
batch_size = 32

# Data Augumentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Rescaling validation & test data
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Automatic Class Detection
train_classes = sorted(os.listdir(os.path.join(preprocessed_dir, 'train')))
print(f"[INFO] Detected classes: {train_classes}")

# Generators
train_generator = train_datagen.flow_from_directory(
    os.path.join(preprocessed_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(preprocessed_dir, 'val'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(preprocessed_dir, 'test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("[INFO] Generators created successfully!")
print(f"Class indices: {train_generator.class_indices}")
