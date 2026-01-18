import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os

def display_prediction_results(image_paths, predictions, model, num_cols=3, img_size=(224, 224)):
    """
    Display images with their predictions in a grid.
    
    Args:
        image_paths: List of paths to images
        predictions: List of prediction probabilities
        model: The model used for predictions
        num_cols: Number of columns in the display grid
        img_size: Size to display images (height, width)
    """
    num_images = len(image_paths)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    plt.figure(figsize=(15, 5 * num_rows))
    
    for i, (img_path, pred) in enumerate(zip(image_paths, predictions)):
        plt.subplot(num_rows, num_cols, i + 1)
        
        # Load and display image (use model input size if available)
        img = image.load_img(img_path, target_size=img_size)
        plt.imshow(img)
        plt.axis('off')
        
        # Determine class and confidence
        if isinstance(pred, np.ndarray):
            pred_value = float(pred.item() if pred.size == 1 else pred[0])
        else:
            pred_value = float(pred)
        class_name = 'Dog' if pred_value > 0.5 else 'Cat'
        confidence = pred_value * 100
        
        # Get filename for title
        filename = os.path.basename(img_path)
        
        # Color based on confidence
        color = 'green' if confidence > 70 else 'orange' if confidence > 50 else 'red'
        
        plt.title(f"{filename}\n{class_name}: {confidence:.2f}%", 
                 color=color, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def calculate_accuracy_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate various accuracy metrics for binary classification.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_pred_proba: Prediction probabilities
    
    Returns:
        Dictionary containing accuracy, precision, recall, F1-score, and confusion matrix components
    """
    # Convert to numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Validate shapes match
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)} elements")
    
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

def plot_confusion_matrix(y_true, y_pred, class_names=['Cat', 'Dog']):
    """
    Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

