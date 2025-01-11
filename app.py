from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import logging
import os

logging.getLogger('tensorflow').setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize the Flask app
app = Flask(__name__)

# Load models
lstm_model = joblib.load("notebooks/lstm_model_store_1.pkl")
random_forest_model = joblib.load("notebooks/random_forest_model_2025-01-10-16-39-33.pkl")
scaler = joblib.load("notebooks/scaler_store_1.pkl")

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict route to handle both models
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        
        store_id = data.get('store_id')
        day_of_week = data.get('day_of_week')
        promo = data.get('promo')

        # Prepare features for the model
        features = np.array([[store_id, day_of_week, promo]])

        # Scale the features using the scaler
        features_scaled = scaler.transform(features)

        # Make predictions using both models
        lstm_prediction = lstm_model.predict(features_scaled)
        rf_prediction = random_forest_model.predict(features_scaled)

        # Return the predictions
        return jsonify({
            'LSTM_Prediction': lstm_prediction.tolist(),
            'Random_Forest_Prediction': rf_prediction.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
