# utils/bpm_estimator.py

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def bandpass_filter(signal, fs, lowcut=1.0, highcut=2.0, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

def estimate_bpm(signal, fs=30.0):
    signal = np.array(signal)
    signal = signal - np.mean(signal)

    filtered = bandpass_filter(signal, fs)

    # Peak detection
    peaks, _ = find_peaks(filtered, distance=fs * 0.4)  # ~40 BPM spacing
    peak_times = np.array(peaks) / fs

    if len(peak_times) < 2:
        return 0.0  # Not enough peaks

    rr_intervals = np.diff(peak_times)
    avg_rr = np.mean(rr_intervals)
    bpm = 60.0 / avg_rr
    return bpm
