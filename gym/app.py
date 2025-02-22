import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_pymongo import PyMongo
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError
import bcrypt
import base64
import io
import json
import pickle
from datetime import datetime, timezone
from PIL import Image
from functools import wraps
from bson.objectid import ObjectId
import sys
from werkzeug.utils import secure_filename
from video_feed import generate_frames, set_exercise, get_results
from flask import Response

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'images', 'profiles')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
app.config["MONGO_URI"] = "mongodb://localhost:27017/fitframe"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and \
           filename.lower().split('.')[-1] in ALLOWED_EXTENSIONS

def get_utc_now():
    return datetime.now(timezone.utc)

# Test MongoDB connection before starting the app
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    client.server_info()  # Will throw an exception if cannot connect
    print("MongoDB connection successful!")
    client.close()
except (ServerSelectionTimeoutError, ConfigurationError) as e:
    print(f"""
ERROR: Could not connect to MongoDB. Please make sure:
1. MongoDB is installed
2. MongoDB service is running (Start MongoDB service from Services app or MongoDB Compass)
3. MongoDB is running on localhost:27017

Error details: {str(e)}
""", file=sys.stderr)
    sys.exit(1)

# Initialize MongoDB
mongo = PyMongo(app)

# Initialize model as None
model = None
try:
    model_architecture_path = 'model/model_architecture.json'
    model_weights_path = 'model/model_weights.pkl'
    
    if os.path.exists(model_architecture_path) and os.path.exists(model_weights_path):
        with open(model_architecture_path, 'r') as f:
            model_json = f.read()
        model = tf.keras.models.model_from_json(model_json)
        with open(model_weights_path, 'rb') as f:
            weights = pickle.load(f)
        for layer, weight_set in zip(model.layers, weights):
            layer.set_weights(weight_set)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("ML model loaded successfully!")
    else:
        print("ML model files not found - running in limited mode")
except Exception as e:
    print(f"Error loading ML model: {str(e)}")
    print("Running in limited mode without ML features")

class_labels = [
    "Bridge", "Camel", "Cat", "Crow", "Extended Side Angle",
    "Forward Bend with Shoulder Opener", "Half-Moon", "Low Lunge",
    "Plank", "Shoulder Stand", "Sphinx", "Upward-Facing Dog",
    "Warrior One", "Warrior Three", "Warrior Two"
]

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_user_profile():
    if 'user_id' in session:
        user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
        if user and 'profile_pic' in user:
            return url_for('static', filename=f"images/profiles/{user['profile_pic']}")
    return None

@app.route('/')
def home():
    return render_template('home.html', 
                         is_authenticated='user_id' in session,
                         profile_pic=get_user_profile())

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            users = mongo.db.users
            login_user = users.find_one({'email': request.form['email']})

            if login_user:
                if bcrypt.checkpw(request.form['password'].encode('utf-8'), login_user['password']):
                    session['user_id'] = str(login_user['_id'])
                    session['username'] = login_user['username']
                    return redirect(url_for('dashboard'))
            
            flash('Invalid email/password combination')
        except Exception as e:
            flash('Database connection error. Please try again later.')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            users = mongo.db.users
            existing_user = users.find_one({'email': request.form['email']})

            if existing_user is None:
                profile_pic = None
                if 'profile_pic' in request.files:
                    file = request.files['profile_pic']
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        unique_filename = f"{get_utc_now().strftime('%Y%m%d%H%M%S')}_{filename}"
                        file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))
                        profile_pic = unique_filename

                hashpass = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())
                users.insert_one({
                    'username': request.form['username'],
                    'email': request.form['email'],
                    'password': hashpass,
                    'profile_pic': profile_pic,
                    'total_points': 0,
                    'daily_points': 0,
                    'last_points_update': get_utc_now(),
                    'exercise_history': []
                })
                return redirect(url_for('login'))
            
            flash('Email address already exists')
        except Exception as e:
            flash('Database connection error. Please try again later.')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
        
        # Reset daily points if it's a new day
        last_update = user.get('last_points_update', get_utc_now())
        current_time = get_utc_now()
        if current_time.date() != last_update.date():
            mongo.db.users.update_one(
                {'_id': ObjectId(session['user_id'])},
                {
                    '$set': {
                        'daily_points': 0,
                        'last_points_update': current_time
                    }
                }
            )
            user['daily_points'] = 0
        
        exercise_history = user.get('exercise_history', [])
        total_points = user.get('total_points', 0)
        daily_points = user.get('daily_points', 0)
        profile_pic = get_user_profile()
        
        rewards = [
            {'name': 'Free Personal Training Session', 'points': 100},
            {'name': 'Premium Workout Plan', 'points': 200},
            {'name': '1-Month Gym Membership', 'points': 500},
            {'name': 'Fitness Equipment Package', 'points': 1000}
        ]
        
        return render_template('dashboard.html', 
                             exercise_history=exercise_history,
                             total_points=total_points,
                             daily_points=daily_points,
                             rewards=rewards,
                             profile_pic=profile_pic)
    except Exception as e:
        flash('Database connection error. Please try again later.')
        return redirect(url_for('home'))

@app.route('/exercise')
@login_required
def exercise():
    return render_template('index.html',
                         class_labels=class_labels,
                         profile_pic=get_user_profile())

@app.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_exercise', methods=['POST'])
@login_required
def start_exercise():
    """Start exercise tracking."""
    data = request.get_json()
    exercise_type = data.get('type', 'bicep_curl')
    duration = data.get('duration', 60)  # Default 60 seconds
    set_exercise(exercise_type, duration)
    return jsonify({'status': 'success'})

@app.route('/reset-counter', methods=['POST'])
@login_required
def reset_counter():
    """Reset exercise counter."""
    try:
        exercise_state = get_exercise_state()
        exercise_state['exercise'].counter = 0
        exercise_state['exercise'].stage = None
        exercise_state['exercise'].perfect_reps = 0
        exercise_state['exercise'].accuracy_score = 0
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_exercise_status')
@login_required
def get_exercise_status():
    """Get current exercise status."""
    results = get_results()
    
    # Update points if accuracy threshold met
    if results['accuracy'] >= 96:
        try:
            user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
            daily_points = user.get('daily_points', 0)
            
            if daily_points < 10:  # Daily points limit
                mongo.db.users.update_one(
                    {'_id': ObjectId(session['user_id'])},
                    {
                        '$inc': {
                            'total_points': 1,
                            'daily_points': 1
                        },
                        '$set': {'last_points_update': get_utc_now()},
                        '$push': {
                            'exercise_history': {
                                'type': results.get('type', 'unknown'),
                                'reps': results['reps'],
                                'accuracy': results['accuracy'],
                                'points_earned': 1,
                                'timestamp': get_utc_now()
                            }
                        }
                    }
                )
        except Exception as e:
            print(f"Error updating points: {str(e)}")
    
    return jsonify(results)

@app.route('/redeem-reward', methods=['POST'])
@login_required
def redeem_reward():
    try:
        data = request.get_json()
        reward_name = data.get('reward')
        points_required = data.get('points')
        
        user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
        total_points = user.get('total_points', 0)
        
        if total_points >= points_required:
            # Deduct points and record redemption
            mongo.db.users.update_one(
                {'_id': ObjectId(session['user_id'])},
                {
                    '$inc': {'total_points': -points_required},
                    '$push': {
                        'redemption_history': {
                            'reward': reward_name,
                            'points': points_required,
                            'timestamp': get_utc_now()
                        }
                    }
                }
            )
            return jsonify({'success': True})
        else:
            return jsonify({
                'success': False,
                'error': 'Insufficient points'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/classify', methods=['POST'])
@login_required
def classify():
    if model is None:
        return jsonify({
            'error': 'ML model not loaded. Running in limited mode.'
        }), 503

    try:
        data = request.get_json()
        image_data = data['image']
        expected_pose = data.get('expected_pose', '')
        
        # Process image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((224, 224))
        img_array = np.array(image)
        
        img_array = img_array.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        predicted_class_name = class_labels[predicted_class_idx]
        
        confidence = float(prediction[0][predicted_class_idx] * 100)
        
        if expected_pose and predicted_class_name != expected_pose:
            confidence = 0.0

        # Update user points if accuracy is >= 96%
        if confidence >= 96:
            try:
                user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
                daily_points = user.get('daily_points', 0)

                # Add point if daily limit not reached
                if daily_points < 10:
                    mongo.db.users.update_one(
                        {'_id': ObjectId(session['user_id'])},
                        {
                            '$inc': {
                                'total_points': 1,
                                'daily_points': 1
                            },
                            '$set': {'last_points_update': get_utc_now()},
                            '$push': {
                                'exercise_history': {
                                    'pose': predicted_class_name,
                                    'accuracy': confidence,
                                    'points_earned': 1,
                                    'timestamp': get_utc_now()
                                }
                            }
                        }
                    )
            except Exception:
                print("Warning: Could not update user points due to database connection error")
        
        return jsonify({
            'prediction': predicted_class_name,
            'confidence': confidence,
            'correct_pose': predicted_class_name == expected_pose if expected_pose else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\nStarting FitFrame application...")
    print(f"Make sure MongoDB is running on localhost:27017")
    print(f"\nAccess the application at:")
    print(f"- Local:   http://localhost:5001")
    print(f"- Network: http://127.0.0.1:5001")
    print(f"\nPress Ctrl+C to quit\n")
    app.run(debug=True, host='0.0.0.0', port=5001)
