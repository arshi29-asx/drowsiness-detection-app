import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import winsound
import threading
import time

st.markdown("""
<style>

/* Main page (keep white) */
.stApp {
    background-color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0e1117;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white;
}

/* Sidebar slider + checkbox labels */
label, .stSlider, .stCheckbox {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
# Page config
st.set_page_config(page_title="Drowsiness Detection", layout="wide")

st.title("🚗 Driver Drowsiness Detection Dashboard")

# Sidebar
st.sidebar.header("⚙️ Controls")
run = st.sidebar.checkbox("Start Camera")

EAR_THRESHOLD = st.sidebar.slider("EAR Threshold", 0.15, 0.35, 0.23)
FRAME_THRESHOLD = st.sidebar.slider("Frame Threshold", 10, 60, 40)

# Layout
col1, col2 = st.columns([2, 1])
chart_placeholder = col2.empty()
frame_placeholder = col1.empty()

# Metrics placeholders
ear_metric = col2.empty()
count_metric = col2.empty()
status_metric = col2.empty()

# Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_points, landmarks, w, h):
    points = []
    for idx in eye_points:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        points.append((x, y))

    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    return (A + B) / (2.0 * C)

# Alarm system
alarm_on = False

def play_alarm():
    global alarm_on
    while True:
        if alarm_on:
            winsound.Beep(1000, 700)
        else:
            time.sleep(0.1)

threading.Thread(target=play_alarm, daemon=True).start()

cap = cv2.VideoCapture(0)

counter = 0
drowsy_count = 0
ear_list = []
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
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
            ear_list.append(ear)

            # keep last 50 values only
            ear_list = ear_list[-50:]
            chart_placeholder.line_chart(ear_list)

            if ear < EAR_THRESHOLD:
                counter += 1

                if counter >= FRAME_THRESHOLD:
                    alarm_on = True

                    if counter == FRAME_THRESHOLD:
                        drowsy_count += 1
            else:
                counter = 0
                alarm_on = False

            # Metrics update
            ear_metric.metric("👁️ EAR", f"{ear:.2f}")
            count_metric.metric("😴 Drowsy Count", drowsy_count)

            if alarm_on:
                status_metric.error("DROWSY 🚨")
            else:
                status_metric.success("NORMAL ✅")

            # Draw landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    frame_placeholder.image(frame, channels="BGR")

cap.release()