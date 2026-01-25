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
import tensorflow as tf
from tensorflow import keras
from keras.utils import load_img, img_to_array
import tensorflow as tf
from keras.models import load_model

import gc
tf.keras.backend.clear_session()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.h5")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_HEIGHT, IMG_WIDTH = 224, 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = keras.models.load_model(MODEL_PATH, compile=False)
print("[INFO] Model loaded successfully!")

class_labels = [
    'MildDemented',
    'ModerateDemented',
    'NonDemented',
    'VeryMildDemented'
]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    filename = None

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return "No file selected"

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            preds = model.predict(img_array)
            pred_class = class_labels[np.argmax(preds)]
            confidence = np.max(preds) * 100

            prediction_text = f"{pred_class} ({confidence:.2f}%)"

    return render_template('index.html', filename=filename, prediction=prediction_text)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
