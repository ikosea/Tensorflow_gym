# Day 6 – Model Evaluation, Regularization, and Explainability
# ----------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import sys

# ------------------------------
# 1. Load the best available model
# ------------------------------
model_paths = [
    r"model/best_model.keras",           # Day 5 best model
    r"../day4_trnsfrcnn/model_epoch_10.keras",
    r"../day4_trnsfrcnn/transfer_cnn_model.h5",
    r"../day4_trnsfrcnn/model_epoch_9.keras"
]

model = None
model_loaded = False
img_size = (224, 224)  # default image size for MobileNetV2

for model_path in model_paths:
    if os.path.exists(model_path):
        try:
            print(f"Trying to load model from {model_path}...")
            model = load_model(model_path)
            print(f"[OK] Model loaded successfully from {model_path}!")
            print(f"Model input shape: {model.input_shape}, output shape: {model.output_shape}")
            img_size = (model.input_shape[1], model.input_shape[2])
            model_loaded = True
            break
        except (ValueError, OSError) as e:
            print(f"[FAILED] Could not load {model_path}: {e}")
            continue

if not model_loaded:
    print("No pre-trained model found. Creating a new MobileNetV2 transfer learning model...")
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
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("Untrained transfer learning model created.")

# ------------------------------
# 2. Load sample images for evaluation
# ------------------------------
sample_dir = r"data/sample_images"
if not os.path.exists(sample_dir):
    raise FileNotFoundError(f"Sample image directory not found: {sample_dir}")

image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(image_files)} sample images.\n")

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# ------------------------------
# 3. Predict and store results
# ------------------------------
predictions = []
ground_truth = []

def get_label_from_filename(filename):
    fname = filename.lower()
    if 'cat' in fname:
        return 0
    elif 'dog' in fname or 'husky' in fname or 'poodle' in fname or 'golden' in fname:
        return 1
    return None

valid_indices = []
for i, img_file in enumerate(image_files):
    label = get_label_from_filename(img_file)
    if label is not None:
        valid_indices.append(i)
        ground_truth.append(label)

        img_path = os.path.join(sample_dir, img_file)
        img_array = load_and_preprocess_image(img_path, img_size)
        pred = model.predict(img_array)
        pred_label = int(pred[0][0] > 0.5)
        predictions.append(pred_label)
        print(f"{img_file}: Prediction = {'Dog' if pred_label else 'Cat'}, Confidence = {pred[0][0]*100:.2f}%")

ground_truth = np.array(ground_truth)
predictions = np.array(predictions)

# ------------------------------
# 4. Evaluation metrics
# ------------------------------
if len(ground_truth) == 0:
    print("No ground truth labels detected. Cannot compute metrics.")
else:
    print("\nConfusion Matrix:")
    cm = confusion_matrix(ground_truth, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Cat','Dog'], yticklabels=['Cat','Dog'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(ground_truth, predictions, target_names=['Cat','Dog']))

    accuracy = np.mean(predictions == ground_truth)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")

# ------------------------------
# 5. Feature map visualization (optional)
# ------------------------------
conv_layers = [layer.output for layer in model.layers if 'conv' in layer.name]
if conv_layers:
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=conv_layers)
    sample_img = load_and_preprocess_image(os.path.join(sample_dir, image_files[0]), img_size)
    activations = activation_model.predict(sample_img)

    plt.figure(figsize=(12,6))
    for i in range(min(6, activations[0].shape[-1])):
        plt.subplot(2,3,i+1)
        plt.imshow(activations[0][0,:,:,i], cmap='viridis')
        plt.axis('off')
    plt.suptitle("Feature Maps from First Conv Layer")
    plt.show()

print("\nDay 6 Evaluation Complete!")
