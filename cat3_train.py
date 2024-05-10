import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Ścieżki do folderów z danymi treningowymi, walidacyjnymi i testowymi
train_dir = 'C:/Users/sylwi/OneDrive/Dokumenty/GitHub/Cat-Detect/train'
validation_dir = 'C:/Users/sylwi/OneDrive/Dokumenty/GitHub/Cat-Detect/validation'
test_dir = 'C:/Users/sylwi/OneDrive/Dokumenty/GitHub/Cat-Detect/test'

# Parametry
batch_size = 32
img_height = 224
img_width = 224

# Obiekt ImageDataGenerator do augmentacji i normalizacji danych
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Przetwarzanie danych treningowych
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary' 
)

# Przetwarzanie danych walidacyjnych
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'  
)

# Przetwarzanie danych testowych
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'  
)

# Definicja modelu do fine-tuningu
base_model = ResNet50V2(weights='imagenet', include_top=False)
base_model.trainable = True

# Dodanie nowej warstwy klasyfikatora
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  
])

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss=BinaryCrossentropy(),  
              metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Ocenianie modelu
loss, accuracy = model.evaluate(test_generator)
print("Test accuracy:", accuracy)

# Zapisanie modelu
model.save("fine_tuned_model.h5")
