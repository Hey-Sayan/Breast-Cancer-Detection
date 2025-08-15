from flask import Flask, request, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load scaler
with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

# Load LSTM model
model = load_model('model.h5')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        features = [float(x) for x in request.form.values()]
        
        # Scale input
        scaled_features = scaler.transform([features])
        
        # Reshape for LSTM (samples, timesteps, features)
        reshaped_features = np.array(scaled_features).reshape((1, scaled_features.shape[1], 1))
        
        # Make prediction
        prediction = model.predict(reshaped_features)
        
        # Assuming output is a probability
        result = "Malignant" if prediction[0][0] > 0.5 else "Benign"
        
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
