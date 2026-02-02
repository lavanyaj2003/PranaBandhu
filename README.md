# Dosha Prediction System

A Flask web application for predicting Ayurvedic Dosha types based on vital signs using machine learning.

## Features

- User registration and authentication
- Dosha prediction using RandomForest model
- Admin dashboard with prediction history
- SQLite database for data storage
- Beautiful responsive UI with Tailwind CSS

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Your Model**
   - Place your `RandomForest.pkl` file in the `flask_app` directory
   - The model should accept input as: [humidity, temperature, spo2, bp]
   - The model should output one of: "Kapha", "Pitta", "Vata"

3. **Run the Application**
   ```bash
   cd flask_app
   python app.py
   ```

4. **Access the Application**
   - Open your browser to `http://localhost:5000`
   - Register a new account or use admin credentials:
     - Email: admin@dosha.com
     - Password: admin123

## Model Integration

Your existing testing code:
```python
import pickle
import numpy as np
mod = pickle.load(open('RandomForest.pkl','rb'))
data = np.array([[60.19,36.5,94.11,147.67]])
prediction = mod.predict(data)[0]
print(prediction)
```

Is integrated into the Flask app in the `predict` route. The system will automatically:
1. Load your model on startup
2. Accept user input for the four vital signs
3. Make predictions using your trained model
4. Store results in the database
5. Display beautiful results to users

## Database Schema

- **users**: id, username, email, password, role, created_at
- **predictions**: id, user_id, humidity, temperature, spo2, bp, prediction, confidence, created_at

## Admin Features

Admin users can:
- View all user predictions
- Monitor system statistics
- See prediction distribution across Dosha types
- Track total users and predictions

## File Structure

```
flask_app/
├── app.py              # Main Flask application
├── dosha.db            # SQLite database (auto-created)
├── RandomForest.pkl    # Your ML model (add this)
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── predict.html
│   ├── result.html
│   ├── dashboard.html
│   └── admin.html
└── README.md
```

## Notes

- The app includes a fallback prediction system if your model file is not found
- All passwords are hashed using SHA-256
- The UI is fully responsive and works on all devices
- Admin account is automatically created on first run