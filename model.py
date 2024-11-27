import tensorflow as tf
from tensorflow import keras as kr
# from kr import layers, models
layers = kr.layers
models = kr.models

def build_c3d_model(input_shape=(16, 112, 112, 3)): 
    model = models.Sequential()
    
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling3D((1, 2, 2)))
    
    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    
    model.add(layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    
    model.add(layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D((2, 2, 2)))  # Changed pooling size
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification
    
    return model

model = build_c3d_model()
model.summary()
