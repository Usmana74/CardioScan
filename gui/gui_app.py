import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
from utils.face_detector import capture_green_signal
from ml.predictor import predict_cardiac_state
import numpy as np
import threading
import time
import platform

# Beep function
if platform.system() == "Windows":
    import winsound
    def beep():
        winsound.Beep(1000, 500)
else:
    def beep():
        print("\a") 

# Paths to model and scaler
MODEL_PATH = "ml/logistic_model.pkl"
SCALER_PATH = "ml/scaler.pkl"

# Corporate color scheme
BG_COLOR = "#A1BAB9"
PRIMARY_COLOR = "#2c3e50"
BUTTON_COLOR = "#d4daca"
TEXT_COLOR = "#272222"
SUCCESS_COLOR = "#27ae60"
WARNING_COLOR = "#f39c12"
DANGER_COLOR = "#c0392b"

color_map = {
    "normal": SUCCESS_COLOR,
    "elevated": WARNING_COLOR,
    "low": DANGER_COLOR
}

def start_scan():
    def perform_scan():
        status_label.config(text="📸 Starting face capture...", fg=PRIMARY_COLOR)
        root.update()

        countdown_seconds = 20

        # Start webcam and face capture
        def capture_with_countdown():
            green_signal = []
            cap = cv2.VideoCapture(0)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            start_time = time.time()
            beep()  # Start beep
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                elapsed = time.time() - start_time
                remaining = int(countdown_seconds - elapsed)
                if remaining < 0:
                    break

                countdown_label.config(text=f"⏳ Time remaining: {remaining}s", fg=PRIMARY_COLOR)
                root.update()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    roi_y_start = y + int(h * 0.1)
                    roi_y_end = y + int(h * 0.25)
                    roi_x_start = x + int(w * 0.3)
                    roi_x_end = x + int(w * 0.7)
                    roi = rgb[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                    green_avg = np.mean(roi[:, :, 1])
                    green_signal.append(green_avg)

            cap.release()
            countdown_label.config(text="")
            beep()  # End beep
            return green_signal

        import cv2
        signal = capture_with_countdown()

        if len(signal) < 10:
            messagebox.showerror("Error", "Not enough data collected. Try again.")
            return

        mean_hr = np.mean(signal)
        std_hr = np.std(signal)
        min_hr = np.min(signal)
        max_hr = np.max(signal)
        range_hr = max_hr - min_hr

        try:
            features = [mean_hr, std_hr, min_hr, max_hr, range_hr]
            prediction = predict_cardiac_state(features, MODEL_PATH, SCALER_PATH)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        mean_label.config(text=f"{mean_hr:.2f} bpm")
        std_label.config(text=f"{std_hr:.2f}")
        result_label.config(text=prediction.upper(), fg=color_map.get(prediction.lower(), TEXT_COLOR))
        status_label.config(text="✅ Scan Complete!", fg=SUCCESS_COLOR)

    threading.Thread(target=perform_scan).start()

# GUI setup
root = tk.Tk()
root.title("CardioScan - Real-time Heart Rate Classifier")
root.attributes('-fullscreen', True)

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

frame = tk.Frame(root, bg=BG_COLOR)
frame.pack(fill=tk.BOTH, expand=True)

header = tk.Frame(frame, bg=BG_COLOR)
header.pack(pady=(30, 10))

logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "logo.png")
logo_img = Image.open(logo_path).resize((100, 100)).convert("RGBA")

mask = Image.new("L", (100, 100), 0)
mask_draw = ImageDraw.Draw(mask)
mask_draw.ellipse((0, 0, 100, 100), fill=255)
logo_img.putalpha(mask)
logo_photo = ImageTk.PhotoImage(logo_img)

logo_label = tk.Label(header, image=logo_photo, bg=BG_COLOR)
logo_label.pack()

title_label = tk.Label(header, text="Cardio Health Scanner", font=("Segoe UI", 32, "bold"), bg=BG_COLOR, fg=PRIMARY_COLOR)
title_label.pack(pady=10)

# Button section
button_frame = tk.Frame(frame, bg=BG_COLOR)
button_frame.pack(pady=10)
tk.Button(button_frame, text="Start Cardio Scan", font=("Segoe UI", 18), bg=BUTTON_COLOR, fg="Teal", padx=30, pady=10, command=start_scan).pack()

# Countdown label
countdown_label = tk.Label(frame, text="", font=("Segoe UI", 20, "bold"), bg=BG_COLOR, fg=PRIMARY_COLOR)
countdown_label.pack(pady=20)

# Results section
result_frame = tk.Frame(frame, bg=BG_COLOR)
result_frame.pack(pady=(10, 30))

tk.Label(result_frame, text="Heart Rate:", font=("Segoe UI", 16), bg=BG_COLOR, fg=TEXT_COLOR).pack()
mean_label = tk.Label(result_frame, text="--", font=("Segoe UI", 16), bg=BG_COLOR, fg=TEXT_COLOR)
mean_label.pack()

tk.Label(result_frame, text="HR Variability:", font=("Segoe UI", 16), bg=BG_COLOR, fg=TEXT_COLOR).pack()
std_label = tk.Label(result_frame, text="--", font=("Segoe UI", 16), bg=BG_COLOR, fg=TEXT_COLOR)
std_label.pack()

tk.Label(result_frame, text="Cardiac State:", font=("Segoe UI", 16), bg=BG_COLOR, fg=TEXT_COLOR).pack()
result_label = tk.Label(result_frame, text="--", font=("Segoe UI", 16), bg=BG_COLOR, fg=TEXT_COLOR)
result_label.pack()

status_label = tk.Label(frame, text="", font=("Segoe UI", 14), bg=BG_COLOR, fg=PRIMARY_COLOR)
status_label.pack(pady=(10, 20))

root.bind('<Escape>', lambda e: root.destroy())

root.mainloop()