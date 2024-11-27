import tensorflow as tf
from preprocess import load_videos_from_directory

def evaluate_model():
    # Load the trained model
    model = tf.keras.models.load_model('results/model.h5')

    # Load test data
    X_test, y_test = load_videos_from_directory('../data/UCF')

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Predict on test data
    y_pred = model.predict(X_test)
    # Convert predictions to binary values (normal vs anomaly)
    y_pred_binary = (y_pred > 0.5).astype(int)
    print(f"Predictions: {y_pred_binary}")

if __name__ == '__main__':
    evaluate_model()
