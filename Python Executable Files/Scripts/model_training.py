import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration
preprocessed_dir = "data/preprocessed" 
img_height, img_width = 224, 224
batch_size = 32
num_classes = 4
epochs = 20
learning_rate = 1e-4

# Folders to save models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results/sample_predictions", exist_ok=True)

# Data Generators
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    os.path.join(preprocessed_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    os.path.join(preprocessed_dir, 'test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Model: VGG16 Transfer Learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('models/best_model.h5', monitor='loss', save_best_only=True, verbose=1)
]

# Training
history = model.fit(
    train_generator,
    epochs=epochs,
    callbacks=callbacks
)

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Loss')
plt.legend()

plt.savefig("results/training_history.png")
plt.show()

# Sample Predictions
test_generator.reset()
x_test, y_test = next(test_generator)

preds = model.predict(x_test)
pred_classes = np.argmax(preds, axis=1)
true_classes = np.argmax(y_test, axis=1)
class_labels = list(train_generator.class_indices.keys())

for i in range(min(5, len(x_test))):
    plt.imshow(x_test[i])
    plt.title(f"True: {class_labels[true_classes[i]]}, Pred: {class_labels[pred_classes[i]]}")
    plt.axis('off')
    plt.savefig(f"results/sample_predictions/pred_{i}.png")
    plt.close()

print("Training and sample predictions complete!")
