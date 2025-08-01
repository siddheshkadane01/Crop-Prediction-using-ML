# 🌾 Farmlytics: Crop Prediction using Machine Learning

A smart agriculture solution that helps farmers predict the most suitable crop to grow based on soil and environmental conditions using machine learning algorithms.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Information](#model-information)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This project implements a machine learning-based crop recommendation system that analyzes various soil and environmental parameters to suggest the most suitable crop for cultivation. The system uses a Random Forest Classifier trained on agricultural data to make accurate predictions.

## ✨ Features

- **Smart Crop Prediction**: Recommends optimal crops based on soil and environmental conditions
- **User-Friendly Web Interface**: Clean and intuitive web application built with Flask
- **Multiple Input Parameters**: Considers 7 key factors for accurate predictions
- **Real-time Results**: Instant crop recommendations
- **Responsive Design**: Works seamlessly across different devices

## 📊 Dataset

The model is trained on a comprehensive crop recommendation dataset containing **2,200+ records** with the following features:

| Parameter | Description | Unit |
|-----------|-------------|------|
| **N** | Nitrogen content in soil | ratio |
| **P** | Phosphorus content in soil | ratio |
| **K** | Potassium content in soil | ratio |
| **Temperature** | Average temperature | °C |
| **Humidity** | Relative humidity | % |
| **pH** | Soil pH value | pH scale |
| **Rainfall** | Average rainfall | mm |

### Supported Crops
The system can predict recommendations for 22 different crops including:
- Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas
- Mothbeans, Mungbean, Blackgram, Lentil, Pomegranate
- Banana, Mango, Grapes, Watermelon, Muskmelon
- Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/siddheshkadane01/Crop-Prediction-using-ML.git
   cd Crop-Prediction-using-ML
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install flask numpy pandas scikit-learn pickle-mixin
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your web browser and navigate to `http://localhost:5000`

## 🎯 Usage

1. **Launch the Web Application**: Start the Flask server using `python app.py`

2. **Input Parameters**: Enter the following soil and environmental conditions:
   - Nitrogen (N) content
   - Phosphorus (P) content  
   - Potassium (K) content
   - Temperature (°C)
   - Humidity (%)
   - Soil pH value
   - Rainfall (mm)

3. **Get Prediction**: Click the "Predict" button to receive crop recommendation

4. **View Results**: The system will display the most suitable crop for the given conditions

## 🤖 Model Information

- **Algorithm**: Random Forest Classifier
- **Training Data**: 2,200+ agricultural records
- **Input Features**: 7 soil and environmental parameters
- **Output**: Crop type recommendation
- **Model Format**: Pickle (.pkl) file for easy deployment

### Model Performance
The Random Forest model was chosen for its:
- High accuracy in handling agricultural data
- Robustness against overfitting
- Ability to handle both numerical features effectively
- Interpretability of feature importance

## 🛠️ Technologies Used

### Backend
- **Python 3.x**: Core programming language
- **Flask**: Web framework for creating the API and web interface
- **scikit-learn**: Machine learning library for model development
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Frontend
- **HTML5**: Structure and content
- **CSS3**: Styling and responsive design
- **JavaScript**: Interactive functionality

### Machine Learning
- **Random Forest Classifier**: Primary prediction algorithm
- **pickle**: Model serialization and deployment

## 📁 Project Structure

```
Crop-Prediction-using-ML/
│
├── app.py                      # Main Flask application
├── model.py                    # Model training script
├── model.pkl                   # Trained machine learning model
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── Dataset/
│   └── Crop_recommendation.csv # Training dataset
│
├── templates/
│   └── index.html              # Main web interface
│
├── static/
│   ├── style.css               # Stylesheet
│   └── farm.jpg                # Background image
│
└── .gitignore                  # Git ignore file
```

## 📸 Screenshots

### Web Interface
![Crop Prediction Interface](static/farm.jpg)
*Clean and intuitive web interface for inputting soil and environmental parameters*

### Sample Prediction
The application provides instant crop recommendations based on the input parameters:
- **Input**: Nitrogen=90, Phosphorus=42, Potassium=43, Temperature=20.8°C, Humidity=82%, pH=6.5, Rainfall=202mm
- **Output**: Recommended Crop: Rice

## 🤝 Contributing

**Siddhesh Kadane**
- GitHub: [@siddheshkadane01](https://github.com/siddheshkadane01)
