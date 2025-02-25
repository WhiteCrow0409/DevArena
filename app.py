import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
import io
import json
import pickle
from PIL import Image

app = Flask(__name__)

try:
    # Load Model (This assumes model_weights.pkl and model_architecture.json are created during training)
    with open('model/model_architecture.json', 'r') as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(model_json)
    with open('model/model_weights.pkl', 'rb') as f:
        weights = pickle.load(f)
    for layer, weight_set in zip(model.layers, weights):
        layer.set_weights(weight_set)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #must recompile after loading weights
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

class_labels = [
    "Bridge", "Camel", "Cat", "Crow", "Extended Side Angle",
    "Forward Bend with Shoulder Opener", "Half-Moon", "Low Lunge",
    "Plank", "Shoulder Stand", "Sphinx", "Upward-Facing Dog",
    "Warrior One", "Warrior Three", "Warrior Two"
]

@app.route('/')
def index():
    return render_template('index.html', class_labels=class_labels)

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        image_data = data['image']
        expected_pose = data.get('expected_pose', '')
        
        # Process image
        image_data = image_data.split(',')[1]  # remove data url prefix
        image_bytes = base64.b64decode(image_data)
        
        # Open and preprocess image using PIL
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((224, 224))
        img_array = np.array(image)
        
        # Convert to float32 and normalize
        img_array = img_array.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Get prediction
        prediction = model.predict(img_array, verbose=0)  # Disable prediction progress bar
        predicted_class_idx = np.argmax(prediction)
        predicted_class_name = class_labels[predicted_class_idx]
        
        # Calculate confidence
        confidence = float(prediction[0][predicted_class_idx] * 100)
        
        # If an expected pose was provided, adjust confidence based on match
        if expected_pose and predicted_class_name != expected_pose:
            confidence = 0.0  # Zero confidence if wrong pose detected
        
        return jsonify({
            'prediction': predicted_class_name,
            'confidence': confidence,
            'correct_pose': predicted_class_name == expected_pose if expected_pose else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)