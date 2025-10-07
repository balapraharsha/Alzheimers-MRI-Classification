# 🧠 Alzheimer’s MRI Classification – Executable Files

This directory contains all executable components of the **Alzheimer’s MRI Classification** project, including the Flask web app, training scripts, models, and results.

---

## 📂 Directory Structure

```bash
Python_Executable_Files/
├── app/ # Flask Web Application
│ ├── app.py # Main Flask app entry point
│ ├── static/ # Static assets (CSS, images, JS)
│ └── templates/ # HTML templates for rendering
│
├── Scripts/ # Supporting Python scripts
│ ├── data_preprocessing.py # Cleans & preprocesses raw images
│ ├── split_data.py # Splits data into train/test/val
│ └── model_training.py # Trains deep learning model
│
├── models/ # Trained model artifacts
│ └── best_model.h5 # Saved Keras model (tracked with Git LFS)
│
├── results/ # Model evaluation outputs
│ └── sample_predictions/ # Example classification outputs
│
├── data/ # Local dataset (ignored in Git)
│ ├── raw/ # Unprocessed MRI scans
│ └── preprocessed/ # Train/Test/Val datasets
│
├── README.md # This documentation
└── requirements.txt # Python depe

```

---

## 📦 Notes
- models/best_model.h5 is tracked using Git LFS to handle large files.

---
