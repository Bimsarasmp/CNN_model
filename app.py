import pickle
import keras
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
import cv2
pip install keras

app = Flask(__name__)

# Load the trained CNN model
custom_objects = {'CategoricalCrossentropy': keras.losses.CategoricalCrossentropy}
cnn_model = load_model('cnn_model.h5', custom_objects=custom_objects, compile=False)


# Load labels dictionary
with open('Models/labels.pickle', 'rb') as handle:
    labels = pickle.load(handle)

# Load ImageDataGenerator configuration
with open('Models/train_datagen.pickle', 'rb') as handle:
    train_datagen = pickle.load(handle)

# Define image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalize pixel values
    return img

# Define image classification route
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Perform classification
    prediction = cnn_model.predict(np.expand_dims(processed_image, axis=0))
    predicted_label = labels[np.argmax(prediction)]

    return jsonify({'predicted_label': predicted_label}), 200

# Define index route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
