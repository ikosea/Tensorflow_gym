import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from src.predict import predict_image, load_and_preprocess_image
from src.utils import display_prediction_results, calculate_accuracy_metrics


model_paths = [
    "model/best_model.keras",  # Day 5 model
    "../day4_trnsfrcnn/model_epoch_10.keras",  # Day 4 latest epoch
    "../day4_trnsfrcnn/transfer_cnn_model.h5",  # Day 4 final model
    "../day4_trnsfrcnn/model_epoch_9.keras",  # Day 4 alternative
]

model = None
model_loaded = False
img_size = (224, 224)  # MobileNetV2 standard size

for model_path in model_paths:
    if os.path.exists(model_path):
        print(f"Trying to load model from {model_path}...")
        try:
            model = load_model(model_path)
            print(f"[OK] Model loaded successfully from {model_path}!")
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
            
            # Determine image size based on model input
            if model.input_shape[1] == 224:
                img_size = (224, 224)
            elif model.input_shape[1] == 150:
                img_size = (150, 150)
            
            print(f"Using image size: {img_size}\n")
            model_loaded = True
            break
        except (ValueError, OSError) as e:
            print(f"[FAILED] Failed to load {model_path}: {e}\n")
            continue

# If no model loaded, create Day 4 style transfer learning model
if not model_loaded:
    print("No pre-trained model found. Creating Day 4 style transfer learning model...")
    print("Using MobileNetV2 with transfer learning approach.\n")
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    img_size = (224, 224)
    print("Transfer learning model created (untrained).")
    print("Note: For best results, train this model or use a pre-trained model from Day 4.\n")

# Sample images directory
sample_dir = "data/sample_images"
image_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(image_files)} sample images\n")
print("=" * 60)

# Process each image and make predictions
predictions = []
image_paths = []

for img_file in image_files:
    img_path = os.path.join(sample_dir, img_file)
    image_paths.append(img_path)
    
    # Load and preprocess image
    img_array = load_and_preprocess_image(img_path, target_size=img_size)
    
    # Make prediction
    prediction = predict_image(model, img_array)
    predictions.append(prediction)
    
    # Display individual prediction
    if isinstance(prediction, np.ndarray):
        pred_value = float(prediction.item() if prediction.size == 1 else prediction[0])
    else:
        pred_value = float(prediction)
    
    print(f"\nImage: {img_file}")
    print(f"Prediction: {'Dog' if pred_value > 0.5 else 'Cat'}")
    print(f"Confidence: {pred_value*100:.2f}%")
    print("-" * 60)

# Display all predictions with images
display_prediction_results(image_paths, predictions, model, img_size=img_size)

# Calculate and display accuracy metrics
print("\n" + "=" * 60)
print("ACCURACY METRICS")
print("=" * 60)

# Automatically determine ground truth labels from filenames
# 0 = Cat, 1 = Dog
def get_label_from_filename(filename):
    """Determine label from filename (cat or dog)"""
    filename_lower = filename.lower()
    if 'cat' in filename_lower:
        return 0  # Cat
    elif 'dog' in filename_lower or 'husky' in filename_lower or 'poodle' in filename_lower or 'golden' in filename_lower or 'running' in filename_lower:
        return 1  # Dog
    else:
        # For numbered files, we can't determine - return None
        return None

ground_truth = []
valid_indices = []
for i, img_file in enumerate(image_files):
    label = get_label_from_filename(img_file)
    if label is not None:
        ground_truth.append(label)
        valid_indices.append(i)

if len(ground_truth) == 0:
    print("Warning: Could not determine ground truth labels from filenames.")
    print("Skipping accuracy metrics calculation.")
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    sys.exit(0)

# Filter predictions to only include valid ones
valid_predictions = [predictions[i] for i in valid_indices]
valid_image_files = [image_files[i] for i in valid_indices]

# Convert ground_truth to numpy array
ground_truth = np.array(ground_truth)

print(f"Evaluating {len(ground_truth)} images with known labels:")
for img_file, label in zip(valid_image_files, ground_truth):
    print(f"  {img_file}: {'Dog' if label == 1 else 'Cat'}")
print()

# Convert predictions to proper format (only for valid images)
pred_array = np.array([
    float(p.item() if isinstance(p, np.ndarray) and p.size == 1 else (p[0] if isinstance(p, np.ndarray) else p))
    for p in valid_predictions
])
predicted_labels = (pred_array > 0.5).astype(int).flatten()

# Ensure both arrays have the same shape
ground_truth = np.array(ground_truth).flatten()
predicted_labels = predicted_labels.flatten()

# Validate shapes match
if len(ground_truth) != len(predicted_labels):
    print(f"Error: Shape mismatch! ground_truth: {ground_truth.shape}, predicted_labels: {predicted_labels.shape}")
    print(f"Number of valid images: {len(valid_predictions)}")
    print(f"Number of ground truth labels: {len(ground_truth)}")
    sys.exit(1)

metrics = calculate_accuracy_metrics(ground_truth, predicted_labels, valid_predictions)

print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
print(f"Precision: {metrics['precision']*100:.2f}%")
print(f"Recall: {metrics['recall']*100:.2f}%")
print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
print(f"\nTrue Positives: {metrics['tp']}")
print(f"True Negatives: {metrics['tn']}")
print(f"False Positives: {metrics['fp']}")
print(f"False Negatives: {metrics['fn']}")

print("\n" + "=" * 60)
print("Evaluation complete!")

