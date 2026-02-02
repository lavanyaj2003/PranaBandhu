from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import hashlib
import pickle
import numpy as np
from datetime import datetime
import os
from functools import wraps
import google.generativeai as genai
genai.configure(api_key='AIzaSyBOzq7o6-tmDja-xH71m1JvzC3b1l7pvug')
gemini_model = genai.GenerativeModel('gemini-2.0-flash')
chat = gemini_model.start_chat(history=[])


app = Flask(__name__)
chat_history = []
app.secret_key = 'your-secret-key-change-in-production'

# Database setup
def init_db():
    conn = sqlite3.connect('dosha.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Predictions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            humidity REAL NOT NULL,
            temperature REAL NOT NULL,
            spo2 REAL NOT NULL,
            bp REAL NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create admin user
    admin_password = hashlib.sha256('admin123'.encode()).hexdigest()
    c.execute('''
        INSERT OR IGNORE INTO users (username, email, password, role) 
        VALUES (?, ?, ?, ?)
    ''', ('admin', 'admin@dosha.com', admin_password, 'admin'))
    
    conn.commit()
    conn.close()

# Load ML model
def load_model():
    try:
        with open('RandomForest.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        # Fallback prediction logic if model file not found
        return None

model = load_model()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash('Admin access required', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect('dosha.db')
        c = conn.cursor()
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        c.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, hashed_password))
        user = c.fetchone()
        
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[4]
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'error')
        
        conn.close()
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return render_template('register.html')
        
        conn = sqlite3.connect('dosha.db')
        c = conn.cursor()
        
        # Check if user exists
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        if c.fetchone():
            flash('Email already registered', 'error')
            conn.close()
            return render_template('register.html')
        
        # Create user
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        c.execute('''
            INSERT INTO users (username, email, password) 
            VALUES (?, ?, ?)
        ''', (username, email, hashed_password))
        
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    
    # Get user's prediction history
    conn = sqlite3.connect('dosha.db')
    c = conn.cursor()
    c.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 10
    ''', (session['user_id'],))
    predictions = c.fetchall()
    conn.close()
    
    return render_template('dashboard.html', predictions=predictions)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            humidity = float(request.form['humidity'])
            temperature = float(request.form['temperature'])
            spo2 = float(request.form['spo2'])
            bp = float(request.form['bp'])
            
           
            data = np.array([[humidity, temperature, spo2, bp]])
            predd = model.predict(data)[0]

            print("{predd}")
            if predd == 0:
                prediction = "Kapha Prakṛti"
                dho="Kapha"
            elif predd == 1:
                prediction = "Pitta Prakṛti"
                dho="Pitta"
            else:
                prediction = "Vāta Prakṛti"
                dho="Vata"
            prmpt =f"{dho} is my dhosa dominant prediciton  so for this u need to recommend following things  1) pathya [diet] recommendation mention brand names and ingredients 2) herbal remidies and how to take it 3) nearby hospital recommendation near hebbal (give exact html format response)"
            gemini_response = chat.send_message(prmpt)
            recommendatoin = gemini_response.text
            recommendatoin = recommendatoin.replace("html", "")
            recommendatoin = recommendatoin.replace("", "")

            prmpt = f"""{dho} is my dhosa dominant prediction. 
            So for this, you need to recommend the following things:
            1) Pathya [diet] recommendation (mention brand names and ingredients)
            2) Herbal remedies and how to take it
            3) Nearby hospital recommendation near Hebbal (give exact HTML format response)

            even if the query is repaeated give the correct response this i attached this in user prediciton page so give correct and in html format with good design
            """

            gemini_response = chat.send_message(prmpt)
            recommendation = gemini_response.text  # get raw HTML (don’t strip tags!)


            
            
            
            
            # Get prediction probabilities if available
            
            probabilities = model.predict_proba(data)[0]
            confidence = max(probabilities)
           
           
            
            # Save prediction to database
            conn = sqlite3.connect('dosha.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions (user_id, humidity, temperature, spo2, bp, prediction, confidence) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session['user_id'], humidity, temperature, spo2, bp, prediction, confidence))
            conn.commit()
            conn.close()
            
            # Get dosha description
            dosha_info = get_dosha_description(dho)
            
            return render_template('result.html', 
                                 prediction=prediction,
                                 confidence=confidence*100,
                                 dosha_info=dosha_info,
                                 recommendation=recommendation,
                                 vitals={'humidity': humidity, 'temperature': temperature, 'spo2': spo2, 'bp': bp})
        
        except ValueError:
            flash('Please enter valid numeric values', 'error')
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'error')
    
    return render_template('predict.html')

@app.route('/admin')
@admin_required
def admin_dashboard():
    conn = sqlite3.connect('dosha.db')
    c = conn.cursor()
    
    # Get all predictions with user info
    c.execute('''
        SELECT p.*, u.username, u.email 
        FROM predictions p 
        JOIN users u ON p.user_id = u.id 
        ORDER BY p.created_at DESC
    ''')
    all_predictions = c.fetchall()
    
    # Get statistics
    c.execute('SELECT COUNT(*) FROM predictions')
    total_predictions = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM users WHERE role = "user"')
    total_users = c.fetchone()[0]
    
    c.execute('''
        SELECT prediction, COUNT(*) 
        FROM predictions 
        GROUP BY prediction
    ''')
    dosha_stats = c.fetchall()
    
    conn.close()
    
    return render_template('admin.html', 
                         predictions=all_predictions,
                         total_predictions=total_predictions,
                         total_users=total_users,
                         dosha_stats=dosha_stats)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

# Helper functions
def fallback_predict(humidity, temperature, spo2, bp):
    """Fallback prediction when model is not available"""
    # Simple rule-based prediction
    if humidity > 65 and temperature < 37 and spo2 > 95:
        return "Kapha"
    elif temperature > 37.5 and bp > 140:
        return "Pitta"
    else:
        return "Vata"

def get_dosha_description(dosha):
    descriptions = {
        'Kapha': {
            'characteristics': 'Earth and Water elements. Stable, calm, and nurturing nature.',
            'recommendations': 'Engage in regular exercise, eat warm and spicy foods, maintain an active lifestyle.',
            'color': '#10B981'
        },
        'Pitta': {
            'characteristics': 'Fire and Water elements. Dynamic, focused, and competitive nature.',
            'recommendations': 'Practice cooling activities, eat cooling foods, manage stress effectively.',
            'color': '#F59E0B'
        },
        'Vata': {
            'characteristics': 'Air and Space elements. Creative, energetic, and flexible nature.',
            'recommendations': 'Follow regular routines, eat warm grounding foods, practice calming activities.',
            'color': '#8B5CF6'
        }
    }
    return descriptions.get(dosha, descriptions[dosha])

if __name__ == '__main__':
    init_db()
    app.run(debug=True)