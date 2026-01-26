import os
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.h5")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


model = load_model(MODEL_PATH)


CLASS_NAMES = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            img = preprocess_image(file_path)
            preds = model.predict(img)[0]

            class_index = np.argmax(preds)
            prediction = CLASS_NAMES[class_index]
            confidence = round(float(np.max(preds)) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
