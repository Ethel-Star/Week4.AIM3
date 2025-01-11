from flask import Flask, request, jsonify, render_template
import joblib
import os
import numpy as np
import pandas as pd

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize the Flask app
app = Flask(__name__)

# Load models
lstm_model = joblib.load("notebooks/lstm_model_store_1.pkl")
random_forest_model = joblib.load("notebooks/random_forest_model_2025-01-10-16-39-33.pkl")
scaler = joblib.load("notebooks/scaler_store_1.pkl")

# Route for home
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data (as JSON)
    data = request.get_json()

    # Assuming the columns used during training were ['store_id', 'day_of_week', 'promo']
    # Create the input data with the same column names as during training
    features = {
        'store_id': [data["store_id"]],
        'day_of_week': [data["day_of_week"]],
        'promo': [data["promo"]]
    }

    # Convert to DataFrame with the appropriate column names
    features_df = pd.DataFrame(features)

    # Make sure the column names match the ones used during training
    expected_columns = ['store_id', 'day_of_week', 'promo']  # Ensure this list matches the order used during model training
    features_df = features_df[expected_columns]

    # Scale the features using the scaler
    try:
        features_scaled = scaler.transform(features_df)
    except ValueError as e:
        return jsonify({"error": f"Scaler error: {str(e)}"}), 400

    # Make prediction using Random Forest model
    try:
        prediction = random_forest_model.predict(features_scaled)
        prediction_value = prediction[0]
    except Exception as e:
        return jsonify({"error": f"Model prediction error: {str(e)}"}), 500

    # Return the prediction as a JSON response
    return jsonify({"prediction": prediction_value})

if __name__ == "__main__":
    app.run(debug=True)
