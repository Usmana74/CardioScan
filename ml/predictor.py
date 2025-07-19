# predictor.py
import joblib
import numpy as np
import os

def predict_cardiac_state(features, model_path, scaler_path):
    """
    features: List or array of numeric values (e.g., [mean_hr, std_hr])
    """
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Reshape input and scale
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Predict
    pred = model.predict(X_scaled)[0]
    label = "Normal" if pred == 0 else "Elevated"
    return label
