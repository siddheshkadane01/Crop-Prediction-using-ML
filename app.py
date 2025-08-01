"""
Farmlytics: Crop Prediction Flask Web Application

A web application that predicts suitable crops based on soil and environmental conditions
using a trained Random Forest machine learning model.

Author: Siddhesh Kadane
"""

import numpy as np
import pickle
import os
from flask import Flask, request, render_template, jsonify

# Create Flask app
app = Flask(__name__)

# Load the trained model
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: model.pkl file not found. Please train the model first using model.py")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    """
    Render the home page with input form
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle crop prediction based on input parameters
    """
    if model is None:
        return render_template("index.html", 
                             prediction_text="Error: Model not loaded. Please contact administrator.")
    
    try:
        # Get form data and convert to float
        features = []
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        for feature_name in feature_names:
            value = request.form.get(feature_name)
            if value is None or value == '':
                raise ValueError(f"Missing value for {feature_name}")
            features.append(float(value))
        
        # Convert to numpy array for prediction
        input_features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        
        # Format the result
        prediction_text = f"🌾 Recommended Crop: {prediction.title()}"
        
        return render_template("index.html", prediction_text=prediction_text)
        
    except ValueError as e:
        error_message = f"❌ Input Error: {str(e)}"
        return render_template("index.html", prediction_text=error_message)
    except Exception as e:
        error_message = f"❌ Prediction Error: An unexpected error occurred. Please try again."
        print(f"Prediction error: {e}")
        return render_template("index.html", prediction_text=error_message)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API endpoint for crop prediction (JSON response)
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        # Validate input data
        required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        features = []
        
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing parameter: {feature}"}), 400
            features.append(float(data[feature]))
        
        # Make prediction
        input_features = np.array(features).reshape(1, -1)
        prediction = model.predict(input_features)[0]
        
        return jsonify({
            "prediction": prediction,
            "input_parameters": {
                "nitrogen": features[0],
                "phosphorus": features[1],
                "potassium": features[2],
                "temperature": features[3],
                "humidity": features[4],
                "ph": features[5],
                "rainfall": features[6]
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template("index.html", 
                         prediction_text="❌ Page not found. Please use the form above."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template("index.html", 
                         prediction_text="❌ Internal server error. Please try again later."), 500

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists("model.pkl"):
        print("Warning: model.pkl not found. Run model.py to train the model first.")
    
    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=5000)