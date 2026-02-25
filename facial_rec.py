"""
Using facial recognition from previous Raspi project so robot can tell the difference between me and other people.
Too slow :( maybe threading could help?
"""

import cv2
import numpy as np
import face_recognition
import pickle
import json
import subprocess
from time import sleep
from gpiozero import Motor, Servo
from ultralytics import YOLO
import pyaudio
from vosk import Model, KaldiRecognizer
import time

MIC_INDEX = 2
RATE = 800

PERSON_FOLLOW = "Jonathan"
BOTTLE_ID = 39 # COCO bottle class

person_face_width = 16 # cm
bottle_width = 6.5     # cm
focal_len = 840

align_tolerance = 80
grab_distance = 27
follow_distance = 55

cv_scaler = 4

# STATES
FOLLOW_BOTTLE = 0
SEARCH_PERSON = 1
FOLLOW_PERSON = 2

# Hopefully to run faster
FACE_SKIP = 5
face_frame_count = 0

state = None
spoken = False

# Load face data
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())

known_face_encodings = data["encodings"]
known_face_names = data["names"]

# HARDWARE
motor1 = Motor(18, 17)
motor2 = Motor(23, 22)

claw = Servo(12, min_pulse_width=0.0006, max_pulse_width=0.0023)
claw.value = 0
sleep(0.5)
claw.detach()

def stop_motors():
    motor1.stop()
    motor2.stop()

def close_claw():
    claw.value = -0.3
    sleep(0.6)
    claw.detach()

def open_claw():
    claw.value = 0
    sleep(0.6)
    claw.detach()

cap_bottle = cv2.VideoCapture(0)
cap_person = cv2.VideoCapture(2)

for cap in (cap_bottle, cap_person):
    cap.set(3, 640)
    cap.set(4, 480)

yolo_model = YOLO("yolov8n_ncnn_model", task="detect")

# Speech recognition
VOSK_MODEL_PATH = "/home/rover/vosk-model-small-en-us-0.15"
vosk_model = Model(VOSK_MODEL_PATH)
rec = KaldiRecognizer(vosk_model, RATE)

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    input_device_index=MIC_INDEX,
    frames_per_buffer=800
)
stream.start_stream()

# FACE PROCESSING
def detect_person(frame):
    resized = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for (top, right, bottom, left), encoding in zip(locations, encodings):
        matches = face_recognition.compare_faces(
            known_face_encodings,
            encoding
        )

        distances = face_recognition.face_distance(
            known_face_encodings,
            encoding
        )

        if len(distances) == 0:
            continue

        best_match = np.argmin(distances)

        if matches[best_match]:
            name = known_face_names[best_match]

            if name == PERSON_FOLLOW:
                # Scale back up
                top *= cv_scaler
                right *= cv_scaler
                bottom *= cv_scaler
                left *= cv_scaler

                width = right - left
                centerx = (left + right) // 2

                if width > 0:
                    distance = (focal_len * person_face_width) / width
                    return True, centerx, distance

    return False, None, None

# MAIN LOOP
print("System ready!")

try:
    while True:
        loop_start = time.time()

        # VOICE CMD (always run)
        data_audio = stream.read(800, exception_on_overflow=False)
        if rec.AcceptWaveform(data_audio):
            result = json.loads(rec.Result())
            text = result.get("text", "").lower()

            if "grab" in text and "bottle" in text:
                print("Voice cmd: FOLLOW_BOTTLE")
                state = FOLLOW_BOTTLE
                spoken = False

        # FSM
        if state == FOLLOW_BOTTLE:
            ret_bottle, frame_bottle = cap_bottle.read()
          
            if not ret_bottle:
                print("Bottle camera error")
                break

            frame_center = frame_bottle.shape[1] // 2

            start_yolo = time.time()

            results = yolo_model.predict(
                frame_bottle,
                imgsz=224,
                conf=0.3,
                verbose=False
            )[0]

            print("YOLO time:", round(time.time() - start_yolo, 3))

            detected = False
            distance = None
            centerx = None

            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id != BOTTLE_ID:
                    continue

                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                width = xmax - xmin
                centerx = (xmin + xmax) // 2

                if width > 0:
                    distance = (focal_len * bottle_width) / width
                    detected = True
                break

            if detected and distance:
                offset = centerx - frame_center

                if distance > grab_distance:
                    if abs(offset) < align_tolerance:
                        motor1.forward(0.3)
                        motor2.forward(0.3)
                    elif offset > 0:
                        motor1.forward(0.4)
                        motor2.backward(0.4)
                    else:
                        motor1.backward(0.4)
                        motor2.forward(0.4)
                else:
                    stop_motors()
                    close_claw()
                    sleep(1)
                    state = SEARCH_PERSON

        elif state == SEARCH_PERSON:
            ret_person, frame_person = cap_person.read()
            if not ret_person:
                print("Person camera error")
                break

            face_frame_count += 1
            if face_frame_count % FACE_SKIP == 0:
                start_face = time.time()
                detected, centerx, distance = detect_person(frame_person)
                print("Face time:", round(time.time() - start_face, 3))

                if detected:
                    stop_motors()
                    state = FOLLOW_PERSON
            else:
                motor1.forward(0.3)
                motor2.backward(0.3)

        elif state == FOLLOW_PERSON:
            ret_person, frame_person = cap_person.read()
            frame_center = frame_person.shape[1] // 2
            face_frame_count += 1

            if face_frame_count % FACE_SKIP == 0:
                start_face = time.time()
                detected, centerx, distance = detect_person(frame_person)
                print("Face time:", round(time.time() - start_face, 3))
            else:
                detected = False

            if detected and distance:
                offset = centerx - frame_center

                if distance > follow_distance:

                    if abs(offset) < align_tolerance:
                        motor1.forward(0.3)
                        motor2.forward(0.3)
                    elif offset > 0:
                        motor1.forward(0.5)
                        motor2.backward(0.5)
                    else:
                        motor1.backward(0.5)
                        motor2.forward(0.5)
                else:
                    stop_motors()
                    open_claw()

                    if not spoken:
                        subprocess.Popen([
                            "espeak",
                            "-a", "200",
                            f"Hello {PERSON_FOLLOW}! Here is your bottle."
                        ])
                        spoken = True
            else:
                state = SEARCH_PERSON

        # Check loop time
        print("Loop time:", round(time.time() - loop_start, 3))

except KeyboardInterrupt:
    print("Shutting down...")

finally:
    stop_motors()
    cap_bottle.release()
    cap_person.release()
    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    p.terminate()
