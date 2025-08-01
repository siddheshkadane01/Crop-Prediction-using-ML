"""
Crop Prediction Model Training Script

This script loads the crop recommendation dataset, trains a Random Forest Classifier,
and saves the trained model as a pickle file for use in the Flask application.

Author: Siddhesh Kadane
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_crop_prediction_model():
    """
    Train a Random Forest model for crop prediction
    """
    print("Loading dataset...")
    # Load the dataset
    data = pd.read_csv('Dataset/Crop_recommendation.csv')
    
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Split the data into features and labels
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    
    print(f"Features shape: {X.shape}")
    print(f"Target classes: {y.unique()}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Create and train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Save the trained model
    print("Saving model...")
    pickle.dump(model, open("model.pkl", "wb"))
    
    print("Model training completed and saved as 'model.pkl'")
    
    # Display feature importance
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    importance = model.feature_importances_
    
    print("\nFeature Importance:")
    for feature, imp in zip(feature_names, importance):
        print(f"{feature}: {imp:.4f}")
    
    return model, accuracy

def test_model_prediction():
    """
    Test the saved model with a sample prediction
    """
    print("\nTesting model with sample data...")
    
    # Load the saved model
    model = pickle.load(open("model.pkl", "rb"))
    
    # Sample test data (N, P, K, temperature, humidity, ph, rainfall)
    sample_data = [[90, 42, 43, 20.8, 82.0, 6.5, 202.9]]
    
    prediction = model.predict(sample_data)
    print(f"Sample prediction: {prediction[0]}")

if __name__ == "__main__":
    # Train the model
    model, accuracy = train_crop_prediction_model()
    
    # Test the model
    test_model_prediction()

