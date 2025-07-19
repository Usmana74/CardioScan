# predictor.py
import joblib
import numpy as np
import os

def predict_cardiac_state(features, model_path, scaler_path):
    # features = [mean_hr, std_hr, min_hr, max_hr, range_hr]
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    return model.predict(X_scaled)[0]

    # Predict
    pred = model.predict(X_scaled)[0]
    label = "Normal" if pred == 0 else "Elevated"
    return label
