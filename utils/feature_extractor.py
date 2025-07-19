# utils/feature_extractor.py

import numpy as np
from scipy.stats import skew, kurtosis
from utils.bpm_estimator import estimate_bpm

def extract_features(signal, fs=30.0):
    signal = np.array(signal)
    signal = signal - np.mean(signal)

    bpm = estimate_bpm(signal, fs)

    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    ptp = np.ptp(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)

    return {
        "bpm": bpm,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "ptp": ptp,
        "skew": skewness,
        "kurtosis": kurt
    }
