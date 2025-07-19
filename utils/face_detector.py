import cv2
import numpy as np
import time

def capture_green_signal(duration=20, roi_type='forehead'):
    green_signal = []
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    start_time = time.time()
    print("[INFO] Starting face capture...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

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

            # Remove these lines for headless capture
            # cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
            # cv2.imshow('Capturing Face', frame)

        # Remove window interruption check
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Collected {len(green_signal)} samples over {duration} seconds.")
    return green_signal
