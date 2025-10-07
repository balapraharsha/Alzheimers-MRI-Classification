"""
Features:
- Upload an MRI image
- Preprocess the image (resize + rescale)
- Predict Alzheimer class using trained VGG16 model
- Display uploaded image and prediction on a styled web page
"""

import os
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = "models/best_model.h5"
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!")

# Hardcoded class labels
class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess & predict
            img_array = preprocess_image(filepath)
            preds = model.predict(img_array)
            pred_class = class_labels[np.argmax(preds)]
            confidence = np.max(preds) * 100

            prediction_text = f"{pred_class} ({confidence:.2f}%)"

    return render_template('index.html', filename=filename, prediction=prediction_text)

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run app
if __name__ == "__main__":
    if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

