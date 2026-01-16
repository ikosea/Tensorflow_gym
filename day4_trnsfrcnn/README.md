# Day 4 – Transfer Learning with CNN

## 📌 Overview
This project demonstrates **Transfer Learning** using a pretrained
Convolutional Neural Network (MobileNetV2) with TensorFlow.

Instead of training a CNN from scratch, a pretrained model is reused
and adapted for a binary image classification task (cat vs dog).

---

## 🧠 What is Transfer Learning?
Transfer Learning allows us to:
- reuse knowledge from a model trained on millions of images
- train faster
- achieve better accuracy with less data

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- MobileNetV2
- Matplotlib

---

## 🧱 Model Architecture
- Pretrained MobileNetV2 (frozen)
- Global Average Pooling
- Dense layer (ReLU)
- Output layer (Sigmoid)

---

## 🚀 How to Run
1. Install dependencies:
```bash
pip install tensorflow matplotlib
