# 💓 CardioScan

CardioScan is a real-time heart rate classification system that uses your webcam to estimate heart rate using remote photoplethysmography (rPPG) and classifies it as **Normal** or **Elevated** using machine learning. It combines computer vision, signal processing, and an intuitive GUI built with Tkinter.

## 📸 Demo

> Real-time contactless heart rate monitoring using only a webcam. *(Add demo screenshot or video here)*

## 🚀 Features

- Real-time pulse detection via webcam  
- rPPG signal extraction using face detection  
- Classification of heart rate: Normal / Elevated  
- Tkinter GUI with live feedback  
- Modular, production-ready codebase  

## 🧠 Technologies Used

- Python  
- OpenCV  
- Scikit-learn  
- Tkinter  
- Pandas, NumPy  

## 🏗️ Project Structure

- `main.py` – Launches the GUI and handles live heart rate prediction  
- `train_model.py` – Trains the ML model on heart rate dataset  
- `dataset/` – Contains training data (e.g., rPPG values and labels)  
- `model/` – Stores trained model files (`.pkl` or `.joblib`)  
- `gui/` – Contains Tkinter UI components (optional split)  
- `utils/` – Helper functions for signal processing and rPPG  
- `requirements.txt` – Python dependencies  
- `README.md` – Project documentation (this file)  

## ⚙️ How to Run

1. **Clone the repository:**

   Open your terminal and run:

git clone https://github.com/Usmana74/CardioScan.git

## 📈 Model & Dataset

- Custom dataset with rPPG-based heart rate readings  
- Labels: Normal (60–90 bpm), Elevated (>90 bpm)  
- Model trained using Scikit-learn classifiers (Logistic Regression) 
