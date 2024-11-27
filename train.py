import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_c3d_model
from preprocess import load_videos_from_directory

import os
os.environ['TF_DISABLE_MKL'] = '1'

def train_model():
    # Load preprocessed data
    X, y = load_videos_from_directory('../data/UCF')

    # Split data into train/test sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Compile and train the model
    model = build_c3d_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

    # Save the trained model
    model.save('results/model.h5')

if __name__ == '__main__':
    train_model()
