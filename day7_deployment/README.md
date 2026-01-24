# Day 7 – Deployment, Explainable AI, and Practical ML Pipelines

## Overview
Day 7 focuses on taking your trained models from theory to practice. You will learn **how to deploy models, automate data pipelines, and create explainable AI outputs** for real-world use. This is critical for turning your ML work into something actionable and understandable by others.

---

## Objectives
By the end of Day 7, you should be able to:

- Deploy a trained ML/CNN model for local or web-based inference  
- Build simple pipelines for preprocessing and prediction  
- Generate interpretable and visual explanations for model predictions  
- Understand ethical considerations and limitations of model decisions  

---

## Topics Covered

### 1. Model Deployment Basics
- Save and load models efficiently (`.keras` or `.h5`)  
- Serve models using:
  - Local scripts for batch predictions  
  - Web APIs (Flask / FastAPI)  
- Test deployed models on new images or datasets  

### 2. Prediction Pipelines
- Automate preprocessing steps:
  - Image resizing & normalization  
  - Label encoding  
- Combine preprocessing + prediction into a single function  
- Batch prediction vs. single-image prediction  

### 3. Explainable AI (XAI)
- Visual explanations for model predictions:
  - Grad-CAM / Heatmaps  
  - Feature importance  
- Communicate **why a model makes certain predictions**  

### 4. Evaluation in Deployment
- Validate predictions with small test datasets  
- Track metrics over time to detect drift  
- Identify misclassifications quickly  

### 5. Ethical & Practical Considerations
- Avoid overreliance on model confidence  
- Document dataset biases  
- Explain model limitations to non-technical stakeholders  

---

## Mini Tasks
1. Wrap your Day 5/6 CNN into a **prediction function** that takes an image path and outputs prediction + confidence.  
2. Save your model in both **`.keras` and `.h5` formats** for compatibility.  
3. Deploy a **local Flask or FastAPI app** to serve your model.  
4. Visualize **Grad-CAM heatmaps** on a few test images.  
5. Track misclassified images and summarize why they failed.  

---

## Expected Outcome
By the end of Day 7, you should be able to:

- Serve your trained model for predictions in a structured pipeline  
- Explain predictions using visual tools like Grad-CAM  
- Identify common deployment pitfalls and how to handle them  
- Move one step closer to **real-world ML applications**  

---

