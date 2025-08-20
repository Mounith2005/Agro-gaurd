import os
import time
import threading
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, session, flash, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ---------------- MongoDB Setup ------------
# 
# ----
client = MongoClient("mongodb://localhost:27017/")  # Update if needed
db = client["plant_disease_db"]
users_collection = db["users"]
feedback_collection = db["feedbacks"]

BASE_FEEDBACK_FOLDER = os.path.join(os.getcwd(), 'feedback')
os.makedirs(BASE_FEEDBACK_FOLDER, exist_ok=True)

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
UPLOAD_DIR = os.path.join(BASE_DIR, 'static/uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Load Model ----------------
try:
    model = load_model(MODEL_PATH)
    print('✅ Model loaded successfully.')
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ---------------- Labels & Solutions ----------------
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

solutions = {
    'Tomato__Bacterial_spot': {
        "disease": "Bacterial Spot",
        "cause": [
            "Caused by the bacterium Xanthomonas campestris.",
            "Spreads through contaminated seeds, soil, and water.",
            "Favored by high humidity and warm temperatures."
        ],
        "symptoms": [
            "Small, dark, water-soaked spots on leaves and fruits.",
            "Spots enlarge over time and may become raised.",
            "Leaves may yellow and drop prematurely, reducing yield."
        ],
        "prevention": [
            "Use certified disease-free seeds.",
            "Practice crop rotation with non-host plants.",
            "Avoid overhead irrigation to keep leaves dry.",
            "Remove and destroy infected plant debris."
        ],
        "remedy": [
            "Apply copper-based fungicides as a preventive measure.",
            "Avoid overhead watering to minimize spread.",
            "Remove and destroy severely infected plants."
        ]
    },
    'Tomato__Early_blight': {
        "disease": "Early Blight",
        "cause": [
            "Caused by the fungus Alternaria solani.",
            "Survives in soil and plant debris.",
            "Spreads via wind and water; accelerated in warm and humid conditions."
        ],
        "symptoms": [
            "Brown concentric rings (target-like) on older leaves.",
            "Leaves may yellow and drop prematurely.",
            "Fruits may develop dark sunken spots."
        ],
        "prevention": [
            "Rotate crops annually to reduce fungal buildup.",
            "Avoid wetting leaves during irrigation.",
            "Plant resistant tomato varieties.",
            "Remove infected plant debris."
        ],
        "remedy": [
            "Apply fungicides such as chlorothalonil or mancozeb.",
            "Prune affected leaves to improve air circulation.",
            "Maintain proper spacing between plants."
        ]
    },
    'Tomato__Late_blight': {
        "disease": "Late Blight",
        "cause": [
            "Caused by Phytophthora infestans (fungus-like organism).",
            "Thrives in cool, wet, and humid conditions.",
            "Spread through water splash and infected plant material."
        ],
        "symptoms": [
            "Large dark, greasy spots on leaves and stems.",
            "White mold growth under leaves in humid conditions.",
            "Rapid plant wilting and fruit rot."
        ],
        "prevention": [
            "Plant resistant tomato varieties.",
            "Avoid overcrowding plants for airflow.",
            "Monitor crops during humid weather.",
            "Remove infected plants immediately."
        ],
        "remedy": [
            "Spray copper-based fungicides or chlorothalonil.",
            "Remove and destroy infected plants.",
            "Avoid working with wet plants."
        ]
    },
    'Tomato__Leaf_Mold': {
        "disease": "Leaf Mold",
        "cause": [
            "Caused by fungus Passalora fulva.",
            "Favors warm, humid environments.",
            "Spreads via water or air and survives on plant debris."
        ],
        "symptoms": [
            "Yellow spots on upper leaf surface.",
            "Olive-green mold on underside of leaves.",
            "Severe infection causes leaf drop and reduced yield."
        ],
        "prevention": [
            "Increase air circulation around plants.",
            "Avoid prolonged leaf wetness.",
            "Prune lower leaves.",
            "Remove infected plant debris."
        ],
        "remedy": [
            "Apply fungicides containing chlorothalonil.",
            "Improve air circulation via spacing and pruning.",
            "Remove severely infected leaves."
        ]
    },
    'Tomato__Septoria_leaf_spot': {
        "disease": "Septoria Leaf Spot",
        "cause": [
            "Caused by fungus Septoria lycopersici.",
            "Survives in soil and plant debris.",
            "Spreads via water; favored by cool, wet conditions."
        ],
        "symptoms": [
            "Small circular spots with dark brown edges and gray centers.",
            "Leaves turn yellow and drop prematurely.",
            "Reduced photosynthesis weakens plant and lowers yield."
        ],
        "prevention": [
            "Remove plant debris at season end.",
            "Use drip irrigation instead of overhead watering.",
            "Rotate crops to non-host plants.",
            "Avoid working with wet plants."
        ],
        "remedy": [
            "Remove affected leaves promptly.",
            "Apply fungicides with mancozeb or chlorothalonil.",
            "Ensure proper spacing and air circulation."
        ]
    },
    'Tomato__Spider_mites_Two_spotted_spider_mite': {
        "disease": "Spider Mites (Two-spotted Spider Mite)",
        "cause": [
            "Infestation by Tetranychus urticae mites.",
            "Thrives in hot, dry conditions.",
            "Spreads via wind or contact with infested plants."
        ],
        "symptoms": [
            "Tiny yellow or white speckles on leaves.",
            "Webbing visible under leaves.",
            "Leaves turn bronze or brown and may drop.",
            "Stunted plant growth in severe infestations."
        ],
        "prevention": [
            "Keep plants well-watered.",
            "Encourage natural predators like ladybugs.",
            "Avoid overuse of insecticides that kill beneficial insects.",
            "Regularly inspect plants for early signs."
        ],
        "remedy": [
            "Apply insecticidal soap or neem oil.",
            "Maintain proper plant hydration.",
            "Remove heavily infested leaves.",
            "Introduce predatory mites if needed."
        ]
    },
    'Tomato__Target_Spot': {
        "disease": "Target Spot",
        "cause": [
            "Caused by fungus Corynespora cassiicola.",
            "Thrives in warm, humid conditions.",
            "Spreads via water, wind, or contaminated tools."
        ],
        "symptoms": [
            "Brown necrotic spots with concentric rings on leaves.",
            "Leaves yellow and drop prematurely.",
            "Fruits develop dark spots reducing quality."
        ],
        "prevention": [
            "Avoid overcrowding plants.",
            "Remove infected plant debris.",
            "Sanitize tools and equipment.",
            "Rotate crops with non-host plants."
        ],
        "remedy": [
            "Spray appropriate fungicides as per instructions.",
            "Remove infected leaves and debris.",
            "Maintain proper spacing between plants."
        ]
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        "disease": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "cause": [
            "Transmitted by whiteflies (Bemisia tabaci).",
            "Virus survives in infected plants and spreads via feeding.",
            "High whitefly populations increase infection risk."
        ],
        "symptoms": [
            "Leaves curl upward and become yellow.",
            "Stunted plant growth with poor fruit set.",
            "Reduced vigor and yield loss."
        ],
        "prevention": [
            "Control whiteflies using traps or insecticides.",
            "Use resistant tomato varieties.",
            "Remove infected plants promptly.",
            "Maintain spacing and avoid excessive nitrogen."
        ],
        "remedy": [
            "Remove infected plants immediately.",
            "Control whiteflies with insecticides.",
            "Monitor remaining plants for early infection signs."
        ]
    },
    'Tomato_healthy': {
        "disease": "Healthy Plant",
        "cause": [
            "No pathogens detected.",
            "Plant shows normal growth and development."
        ],
        "symptoms": [
            "Leaves are green and uniform.",
            "Stem is strong and upright.",
            "Overall plant appears vigorous and healthy."
        ],
        "prevention": [
            "Maintain proper watering and fertilization.",
            "Monitor regularly for pests and diseases.",
            "Ensure adequate spacing and airflow.",
            "Use certified seeds."
        ],
        "remedy": [
           "No treatment needed.\n",
            "Keep monitoring for early detection of issues.",
            "Maintain good agricultural practices."
        ]
    }
}


lock = threading.Lock()

# ---------------- Helper Functions ----------------
def cleanup_uploads(max_age=86400):
    now = time.time()
    for f in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, f)
        if os.path.isfile(fpath) and os.stat(fpath).st_mtime < now - max_age:
            os.remove(fpath)

def get_result(image_path):
    if model is None:
        return "Model not loaded", 0.0, "Prediction unavailable."
    try:
        img = load_img(image_path, target_size=(225, 225))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        predictions = model.predict(x, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_label = labels[predicted_index]
        confidence = float(predictions[predicted_index])
        solution = solutions.get(predicted_label, "No solution available.")
        return predicted_label, confidence, solution
    except Exception as e:
        return str(e), 0.0, "Error in prediction."
    

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- Routes ----------------

@app.route('/')
def home():
    # Check if user is logged in
    if 'role' in session:
        if session['role'] == 'admin':
            return redirect('/admin_dashboard')  # Admin goes to admin dashboard
        elif session['role'] == 'user':
            return render_template('index.html', username=session['username'])  # User sees home
    # If no session, show index page as the first page
    return render_template('index.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.form['username'].strip()
        password = request.form['password']
        role = request.form.get('role', 'user')  # default role

        # Hardcoded admin login
        if username == "manivel" and password == 'manivel@12' and role == 'admin':
            session['username'] = username
            session['role'] = 'admin'
            flash(f"Welcome {username}!", "success")
            return redirect('/admin_dashboard')

        # MongoDB user login
        user = users_collection.find_one({"username": username})
        if user and user['password'] == password and user.get('role', 'user') == role:
            session['username'] = user['username']
            session['role'] = user.get('role', 'user')
            flash(f"Welcome {username}!", "success")
            return redirect('/')  # regular user dashboard

        else:
            flash("Invalid credentials or role", "danger")
            return redirect('/login')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == "POST":
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        role = request.form.get('role', 'user')  # default role as 'user'

        # Validation
        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return redirect('/register')

        if users_collection.find_one({"username": username}):
            flash("Username already exists", "danger")
            return redirect('/register')

        if users_collection.find_one({"email": email}):
            flash("Email already registered", "danger")
            return redirect('/register')

        # Hash password for security (optional, recommended)
        # hashed_password = generate_password_hash(password)

        # Insert into MongoDB
        user_doc = {
            "username": username,
            "email": email,
            "phone": phone,
            "password": password,  # or hashed_password
            "role": role           # corrected key
        }
        users_collection.insert_one(user_doc)

        flash("Registration successful. Please login.", "success")
        return redirect('/login')

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


# ---------------- Admin Dashboard ----------------
@app.route('/admin_dashboard')
def admin_dashboard():
    if 'role' in session and session['role'] == 'admin':
        feedbacks = list(feedback_collection.find())
        return render_template('admin_dashboard.html', feedbacks=feedbacks)
    return redirect('/login')


# ---------------- User Feedback ----------------
@app.route('/user_feedback', methods=['GET', 'POST'])
def user_feedback():
    if request.method == "POST":
        feedback_text = request.form['feedback']
        feedback_collection.insert_one({
            "username": session.get('username', 'Anonymous'),
            "feedback": feedback_text,
            "timestamp": time.time()
        })
        flash("Feedback submitted successfully!", "success")
        return redirect('/user_feedback')

    return render_template('user_feedback.html', username=session.get('username', 'Guest'))

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    username = session.get('username', 'Anonymous')
    label = request.form.get('label', '')
    feedback_text = request.form.get('feedback', '')
    timestamp = time.time()

    # Create user-specific folder
    user_folder = os.path.join(BASE_FEEDBACK_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)

    # Handle file upload
    uploaded_file = request.files.get('file')
    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(user_folder, filename)
        uploaded_file.save(file_path)
    else:
        filename = None
        file_path = None

    # Save feedback as text file in user folder
    feedback_filename = f"feedback_{int(timestamp)}.txt"
    feedback_file_path = os.path.join(user_folder, feedback_filename)
    with open(feedback_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Label: {label}\n")
        f.write(f"Feedback: {feedback_text}\n")
        f.write(f"Timestamp: {time.ctime(timestamp)}\n")

    # Insert into MongoDB
    feedback_doc = {
        "username": username,
        "filename": filename,
        "file_path": file_path,
        "feedback_file_path": feedback_file_path,
        "label": label,
        "feedback": feedback_text,
        "timestamp": timestamp
    }

    feedback_collection.insert_one(feedback_doc)
    flash("Feedback submitted successfully!", "success")
    return redirect('/')

# ---------------- Detection ----------------
@app.route('/detect')
def detect_page():
    if 'username' not in session:
        return redirect('/login')
    return render_template('detect.html')


@app.route('/predict', methods=['POST'])
def predict():
    cleanup_uploads()

    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file
    filename = f"{int(time.time())}_{secure_filename(file.filename)}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    # Get prediction
    predicted_label, confidence, solution = get_result(file_path)

    # Ensure solution is a dictionary (in case of errors)
    if not isinstance(solution, dict):
        solution = {"disease": predicted_label, "remedy": solution}

    # Render template with all details
    return render_template(
        'detection.html',
        filename=filename,
        label=predicted_label,
        confidence=confidence,
        solution=solution
    )

# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(debug=True)
