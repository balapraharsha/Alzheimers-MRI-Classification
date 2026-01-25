import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import MobileNetV2  
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


preprocessed_dir = "data/preprocessed" 
img_height, img_width = 224, 224
batch_size = 32
num_classes = 4
epochs = 20
learning_rate = 1e-4

os.makedirs("models", exist_ok=True)
os.makedirs("results/sample_predictions", exist_ok=True)

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    os.path.join(preprocessed_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('models/best_model.h5', monitor='loss', save_best_only=True, verbose=1)
]

model.fit(train_generator, epochs=epochs, callbacks=callbacks)
print("Training complete! Model saved to models/best_model.h5")