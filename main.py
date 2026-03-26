import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import winsound
import os
# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Calculate EAR
def calculate_ear(eye_points, landmarks, frame_w, frame_h):
    points = []
    for idx in eye_points:
        x = int(landmarks[idx].x * frame_w)
        y = int(landmarks[idx].y * frame_h)
        points.append((x, y))

    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    return (A + B) / (2.0 * C)

# 🔊 Alarm control


def play_alarm():
    global alarm_on
    while True:
        if alarm_on:
            os.system("printf '\\a'")  # simple beep
        else:
            time.sleep(0.1)

# Start alarm thread once
threading.Thread(target=play_alarm, daemon=True).start()

# Start webcam
cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20

counter = 0
drowsy_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            left_ear = calculate_ear(LEFT_EYE, landmarks, w, h)
            right_ear = calculate_ear(RIGHT_EYE, landmarks, w, h)

            ear = (left_ear + right_ear) / 2.0

            # 👁️ Draw eye landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # Show EAR
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 🔴 Drowsy logic
            if ear < EAR_THRESHOLD:
                counter += 1

                if counter >= FRAME_THRESHOLD:
                    cv2.putText(frame, "DROWSY!", (200, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                    alarm_on = True

                    # Count only once per event
                    if counter == FRAME_THRESHOLD:
                        drowsy_count += 1

            else:
                counter = 0
                alarm_on = False

    # 📊 Show drowsy count
    cv2.putText(frame, f"Drowsy Count: {drowsy_count}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 🎨 Status UI box
    status = "NORMAL"
    color = (0, 255, 0)

    if alarm_on:
        status = "DROWSY"
        color = (0, 0, 255)

        # Warning text
        cv2.putText(frame, "WAKE UP!", (200, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.rectangle(frame, (20, 80), (260, 140), color, -1)

    cv2.putText(frame, status, (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
