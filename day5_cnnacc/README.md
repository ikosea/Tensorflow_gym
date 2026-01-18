# Day 5 – CNN Accuracy Evaluation

## 📌 Overview
This project demonstrates how to **evaluate the accuracy** of a trained Convolutional Neural Network (CNN) model for image classification. You'll learn how to:

- Load a pre-trained CNN model
- Make predictions on new images
- Calculate accuracy metrics (accuracy, precision, recall, F1-score)
- Visualize prediction results

The project uses a binary classification model (Cat vs Dog) to demonstrate accuracy evaluation techniques.

---

## 🧠 Concepts Learned
- Model evaluation and accuracy metrics
- Precision, Recall, and F1-Score
- Confusion matrix components (TP, TN, FP, FN)
- Making predictions on new data
- Visualizing model predictions

---

## 🛠️ Technologies Used
- **Python**
- **TensorFlow / Keras** - Deep learning framework
- **NumPy** - Numerical operations
- **Matplotlib** - Visualization
- **Pillow (PIL)** - Image processing

---

## 📊 Accuracy Metrics Explained

### Accuracy
The proportion of correct predictions out of all predictions.
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision
The proportion of positive predictions that are actually correct.
```
Precision = TP / (TP + FP)
```

### Recall (Sensitivity)
The proportion of actual positives that were correctly identified.
```
Recall = TP / (TP + FN)
```

### F1-Score
The harmonic mean of precision and recall.
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### Confusion Matrix Components
- **TP (True Positives)**: Correctly predicted as positive
- **TN (True Negatives)**: Correctly predicted as negative
- **FP (False Positives)**: Incorrectly predicted as positive
- **FN (False Negatives)**: Incorrectly predicted as negative

---

## 📁 Project Structure
```
day5_cnnacc/
├── main.py                 # Main evaluation script
├── model/
│   └── best_model.keras    # Pre-trained CNN model
├── data/
│   └── sample_images/      # Test images for evaluation
│       ├── cats.jpeg
│       ├── dog_running.jpeg
│       ├── goldenretriever.jpeg
│       ├── husky.jpeg
│       └── poodle.jpeg
├── src/
│   ├── __init__.py
│   ├── predict.py          # Prediction functions
│   └── utils.py            # Utility functions for metrics and visualization
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install tensorflow numpy matplotlib pillow
```

Or install from the main requirements file:
```bash
pip install -r ../requirements.txt
```

### 2. Run the Evaluation Script
```bash
python main.py
```

---

## 📝 What the Script Does

1. **Loads the pre-trained model** from `model/best_model.keras`
2. **Processes sample images** from `data/sample_images/`
3. **Makes predictions** on each image
4. **Displays predictions** with confidence scores
5. **Calculates accuracy metrics** (accuracy, precision, recall, F1-score)
6. **Visualizes results** with images and predictions

---

## 📈 Expected Output

The script will:
- Display model information (input/output shapes)
- Show predictions for each image with confidence scores
- Display a grid of images with their predictions
- Print accuracy metrics:
  - Overall Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion matrix breakdown (TP, TN, FP, FN)

---

## 🎯 Key Functions

### `load_and_preprocess_image(img_path, target_size)`
Loads and preprocesses an image for model input.

### `predict_image(model, img_array)`
Makes a prediction on a preprocessed image.

### `display_prediction_results(image_paths, predictions, model)`
Visualizes images with their predictions in a grid layout.

### `calculate_accuracy_metrics(y_true, y_pred, y_pred_proba)`
Calculates accuracy, precision, recall, F1-score, and confusion matrix components.

---

## 💡 Tips

- **Confidence Threshold**: The model uses 0.5 as the threshold for binary classification
- **Image Preprocessing**: Images are resized to 150x150 and normalized to [0, 1]
- **Ground Truth Labels**: Update the `ground_truth` array in `main.py` with actual labels for accurate metrics

---

## 🔍 Understanding the Results

- **High Accuracy**: Model correctly classifies most images
- **High Precision**: When the model predicts "Dog", it's usually correct
- **High Recall**: The model finds most actual dogs in the dataset
- **Balanced F1-Score**: Indicates a good balance between precision and recall

---

## 📚 Next Steps

- Experiment with different confidence thresholds
- Evaluate on a larger test dataset
- Compare metrics across different models
- Implement ROC curve and AUC metrics
- Add support for multi-class classification

---

## 🐛 Troubleshooting

**Model not found?**
- Ensure `model/best_model.keras` exists
- Check the model path in `main.py`

**Image loading errors?**
- Verify images are in `data/sample_images/`
- Check image file formats (jpg, jpeg, png)

**Import errors?**
- Make sure all dependencies are installed
- Check Python version (TensorFlow requires Python 3.8+)

---

## 📖 Related Projects
- **Day 3**: Building a CNN from scratch
- **Day 4**: Transfer learning with pre-trained models

---

**Happy Learning! 🚀**
