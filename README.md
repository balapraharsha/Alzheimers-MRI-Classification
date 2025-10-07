# 🧠 Alzheimer’s MRI Classification 🩺

AI-powered web application for automated staging of dementia severity from $T_1$-weighted MRI scans.  
Classifies MRI slices into **Non-Demented**, **Very Mild Demented**, **Mild Demented**, and **Moderate Demented** using **VGG16 transfer learning**.

---

## 🗂️ Project Structure
```bash

Alzheimers_MRI_Classification/
├── Python_Executable_Files/         # All project components
│   ├── app/                         # Web Deployment Component
│   │   ├── app.py
│   │   ├── static/
│   │   │   └── style.css
│   │   └── templates/
│   │       └── index.html
│   ├── Scripts/                     # Supporting Scripts
│   │   ├── data_preprocessing.py
│   │   ├── split_data.py
│   │   └── model_training.py
│   ├── models/                      # Trained Model Artifacts
│   │   └── best_model.h5
│   ├── results/                     # Model Evaluation Outputs
│   │   └── sample_predictions/
│   ├── data/                        # Data Component
│   │   ├── raw/                     # Original, unprocessed images
│   │   │   ├── Mild Demented/
│   │   │   ├── Moderate Demented/
│   │   │   ├── Non Demented/
│   │   │   └── Very Mild Demented/
│   │   └── preprocessed/            # Cleaned and processed images
│   │       ├── train/
│   │       │   ├── Mild Demented/
│   │       │   ├── Moderate Demented/
│   │       │   ├── Non Demented/
│   │       │   └── Very Mild Demented/
│   │       ├── test/
│   │       │   ├── Mild Demented/
│   │       │   ├── Moderate Demented/
│   │       │   ├── Non Demented/
│   │       │   └── Very Mild Demented/
│   │       └── val/
│   │           ├── Mild Demented/
│   │           ├── Moderate Demented/
│   │           ├── Non Demented/
│   │           └── Very Mild Demented/
│   └── README.md                    # Internal doc for executables
│   
├── README.md                        # Root-level project documentation
├── Project Documentation.pdf        # Project documentation pdf
└── requirements.txt                 # Root-level environment dependencies



```

## ✨ Features
- Automated dementia classification from MRI scans  
- Confidence scores & probability distribution  
- Interactive web app for real-time predictions  
- Training visualization & sample prediction saving  

---

## 🛠️ Tech Stack
- Python 3.11  
- TensorFlow / Keras  
- Flask (Web Framework)  
- Tailwind CSS  
- Matplotlib, Numpy

---

## 🚀 How to Run

**Clone the repository**
```bash
git clone https://github.com/yourusername/Alzheimers_MRI_Classification.git
cd Alzheimers_MRI_Classification
```

**Create virtual environment & activate**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Place dataset**
- Dataset should be under data/preprocessed/ with train & test folders for each class.


**Train the model (optional if best_model.h5 exists)**
```bash
python Scripts/model_training.py
```

**Run the Flask app**
```bash
python app.py
```

**Open in browser**
- As given local host


---

## 🖥️ Functionalities
- Upload MRI scan (JPG/PNG)
- View predicted class & confidence score
- Visualize probability distribution
- Clinical interpretation notes
- Sample predictions saved locally

---

## 📊 Model Details
- Architecture: VGG16 Transfer Learning (base frozen)
- Input: 224 x 224 x 3 RGB MRI slice
- Output: Softmax over 4 classes
- Optimizer: Adam (lr=1e-4)
- Loss: Categorical Crossentropy
- Epochs: 20, Batch Size: 32
- Expected Validation Accuracy: ~97%

---
## 🗃️ Dataset Overview
- Public Alzheimer’s MRI datasets (OASIS, ADNI subsets)
- Preprocessed into 2D slices, resized to 224x224 pixels
- Directory structure
```bash
train/
├── Non-Demented/
├── Very Mild Demented/
├── Mild Demented/
└── Moderate Demented/
test/
├── Non-Demented/
├── Very Mild Demented/
├── Mild Demented/
└── Moderate Demented/
```

---

## 📈 Insights Summary
- Deep learning effectively detects structural brain changes
- High confidence in Non-Demented vs Severe Demented detection
- Model useful for early-stage dementia screening

---

## 🌟 Highlights
- Full-stack AI project with ML + Web App
- Real-time predictions with confidence & probability bars
- User-friendly interface for researchers & developers
- Training visualization & sample prediction outputs

---

## 👨‍💻 Developed By

**Bala Praharsha .M**  
📧 [balapraharsha.m@gmail.com]  
🔗 [LinkedIn](https://linkedin.com/in/mannepalli-bala-praharsha) | [GitHub](https://github.com/balapraharsha)  

---

## 💖 Show Some Love
Enjoying this project? Give it a **star** ⭐ on GitHub!  
Contributions, suggestions, and forks are always welcome.

