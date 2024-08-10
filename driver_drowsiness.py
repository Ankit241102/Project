import dlib
import cv2
import numpy as np
from imutils import face_utils
import pygame
import threading
import time
import pyttsx3  

pygame.mixer.init()

engine = pyttsx3.init()
lock = threading.Lock()

# Load face detector and predictor, uses dlib shape predictor file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\hp\Desktop\Ankit_minor _project\Driver-Drowsiness-Detection-master\Driver-Drowsiness-Detection-master\shape_predictor_68_face_landmarks.dat")

# Status marking for current state
drowsy = 0
status = ""
color = (0, 0, 0)
alarm_playing = False
eye_closed_time = 0.0  # Variable to track eye closure time

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.20:
        return 2
    
        
    else:
        return 1

def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        pygame.mixer.music.load(r"C:\Users\hp\Desktop\project_minor\Driver-Drowsiness-Detector by Ankit kaurav\Driver-Drowsiness-Detector by Ankit kaurav\alert.wav")
        pygame.mixer.music.play(-1)  # Loop the alarm sound
        # engine.say("Warning! Ankit You are drowsy.")
        engine.runAndWait()

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False

def update_status(eye_status):
    global drowsy, status, color, eye_closed_time
    
    if eye_status == 1:
        eye_closed_time = 0.0
        drowsy += 1
        if drowsy > 0:
            status = "Drowsy !"
            color = (0, 0, 255)
            if not alarm_playing and (time.time() - eye_closed_time >= 6):
                threading.Thread(target=play_alarm).start()
    else:
        drowsy = 0
        status = "Active :)"
        color = (0, 255, 0)
        if alarm_playing:
            stop_alarm()
            eye_closed_time = time.time()

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Reduce frame size to speed up processing
    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    # Initialize face_frame outside the loop
    face_frame = frame.copy()

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
                
        eye_status = min(left_blink, right_blink)
        update_status(eye_status)

        # Display status
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        

        for (x, y) in landmarks:
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
