import os
import time
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
FEEDBACK_FILE = os.path.join(BASE_DIR, 'feedback.txt')

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, 'w') as f:
        f.write("filename\tlabel\tfeedback\n")

# Load model
try:
    model = load_model(MODEL_PATH)
    print('Model loaded successfully. Check http://127.0.0.1:5000/')
except Exception as e:
    print(f"Error loading model: {e}")

# Labels dictionary
labels = {
    0: 'Tomato__Bacterial_spot',
    1: 'Tomato__Early_blight',
    2: 'Tomato__Late_blight',
    3: 'Tomato__Leaf_Mold',
    4: 'Tomato__Septoria_leaf_spot',
    5: 'Tomato__Spider_mites_Two_spotted_spider_mite',
    6: 'Tomato__Target_Spot',
    7: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    8: 'Tomato_healthy'
}

# Prediction function
def get_result(image_path):
    try:
        img = load_img(image_path, target_size=(225, 225))
        x = img_to_array(img)
        x = x.astype('float32') / 255.0
        x = np.expand_dims(x, axis=0)
        predictions = model.predict(x, verbose=0)[0]
        predicted_index = np.argmax(predictions)
        predicted_label = labels[predicted_index]
        confidence = float(predictions[predicted_index])
        return predicted_label, confidence
    except Exception as e:
        return str(e), 0.0

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    filename = f"{int(time.time())}_{secure_filename(file.filename)}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    predicted_label, confidence = get_result(file_path)
    return jsonify({
        'filename': filename,
        'label': predicted_label,
        'confidence': confidence
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    filename = data.get('filename')
    label = data.get('label')
    feedback_value = data.get('feedback')

    if not filename or not label or not feedback_value:
        return jsonify({'status': 'error', 'message': 'Missing data'}), 400

    with open(FEEDBACK_FILE, 'a') as f:
        f.write(f"{filename}\t{label}\t{feedback_value}\n")

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
