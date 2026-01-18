import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for model prediction.
    
    Args:
        img_path: Path to the image file
        target_size: Target size for the image (height, width)
    
    Returns:
        Preprocessed image array ready for model input
    """
    # Load image
    img = image.load_img(img_path, target_size=target_size)
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match model input shape (batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    
    return img_array

def predict_image(model, img_array):
    """
    Make a prediction on a preprocessed image.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array
    
    Returns:
        Prediction probability (0 = Cat, 1 = Dog for binary classification)
    """
    prediction = model.predict(img_array, verbose=0)
    return prediction

def predict_batch(model, img_paths, target_size=(224, 224)):
    """
    Make predictions on a batch of images.
    
    Args:
        model: Trained Keras model
        img_paths: List of image file paths
        target_size: Target size for images
    
    Returns:
        Array of predictions
    """
    predictions = []
    for img_path in img_paths:
        img_array = load_and_preprocess_image(img_path, target_size)
        prediction = predict_image(model, img_array)
        predictions.append(prediction)
    
    return np.array(predictions)

