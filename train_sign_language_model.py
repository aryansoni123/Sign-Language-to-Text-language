# train_sign_language_model.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define paths
dataset_dir = 'dataset'  # Update this path

# Function to create a dataframe from directory
def dataframe_from_directory(directory):
    categories = []
    filenames = []
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                if os.path.isfile(file_path):
                    categories.append(category)
                    filenames.append(file_path)
    df = pd.DataFrame({
        'filename': filenames,
        'class': categories
    })
    return df

# Create dataframe from dataset directory
df = dataframe_from_directory(dataset_dir)

# Remove rows with missing files
df = df[df['filename'].apply(os.path.exists)]

# Image data generator
datagen = ImageDataGenerator(rescale=0.2, validation_split=0.2)

# Training data generator
train_generator = datagen.flow_from_dataframe(
    df,
    x_col='filename',
    y_col='class',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
validation_generator = datagen.flow_from_dataframe(
    df,
    x_col='filename',
    y_col='class',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Get the number of classes
num_classes = len(train_generator.class_indices)

# Build the model
model = Sequential([
    Input(shape=(64, 64, 3)),  # Explicitly define the input layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
model.save('sign_language_model.h5')
