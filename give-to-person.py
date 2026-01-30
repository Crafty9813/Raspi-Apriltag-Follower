"""
Uses 1 camera to find and grab a bottle (or other object) and another one to find a person and follow them until a certain distance away, and then opens the claw. 
Robot rotates in place to find the person after picking up the bottle.
"""

import cv2
from ultralytics import YOLO
from gpiozero import Motor, Servo
from time import sleep

model = YOLO("yolov8n_ncnn_model", task="detect")

PERSON_ID = 0
BOTTLE_ID = 39

person_width = 30 # cm
bottle_width = 6.5 # cm
focal_len = 840

align_tolerance = 80
grab_distance = 20 # cm
follow_distance = 60 # cm

# States
FOLLOW_BOTTLE = 0
SEARCH_PERSON = 1
FOLLOW_PERSON = 2

state = FOLLOW_BOTTLE

motor1 = Motor(18, 17)
motor2 = Motor(23, 22)

claw = Servo(12, min_pulse_width=0.0006, max_pulse_width=0.0023)
claw.value = 0
sleep(0.5)
claw.detach()

cap_bottle = cv2.VideoCapture(2)
cap_person = cv2.VideoCapture(0)

for cap in (cap_bottle, cap_person):
    cap.set(3, 640)
    cap.set(4, 480)

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

while True:
    ret_bottle, frame_bottle = cap_bottle.read()
    ret_person, frame_person = cap_person.read()

    if not ret_bottle or not ret_person:
        print("Camera read failed")
        break

    # Camera selection
    if state == FOLLOW_BOTTLE:
        frame = frame_bottle
        target_id = BOTTLE_ID
        target_width = bottle_width
    else:
        frame = frame_person
        target_id = PERSON_ID
        target_width = person_width

    frame_center = frame.shape[1] // 2
    detected = False
    distance = None
    centerx = None

    results = model.predict(frame, imgsz=320, conf=0.25, verbose=False)[0]

    for bb in results.boxes:
        cls_id = int(bb.cls[0])
        if cls_id != target_id:
            continue

        xmin, ymin, xmax, ymax = map(int, bb.xyxy[0])
        pixel_width = xmax - xmin
        centerx = (xmin + xmax) // 2

        if pixel_width > 0:
            distance = (focal_len * target_width) / pixel_width

        detected = True
        break

    # State machine
    if state == FOLLOW_BOTTLE:
        if detected and distance is not None:
            offset = centerx - frame_center

            if distance > grab_distance:
                if abs(offset) < align_tolerance:
                    motor1.forward(0.2)
                    motor2.forward(0.2)
                elif offset > 0:
                    motor1.forward(0.4)
                    motor2.backward(0.4)
                else:
                    motor1.backward(0.4)
                    motor2.forward(0.4)
            else:
                stop_motors()
                sleep(0.5)
                close_claw()
                sleep(1)
                state = SEARCH_PERSON
        else:
            stop_motors()

    elif state == SEARCH_PERSON:
        motor1.forward(0.4)
        motor2.backward(0.4)

        if detected and target_id == PERSON_ID:
            stop_motors()
            state = FOLLOW_PERSON

    elif state == FOLLOW_PERSON:
        if detected and distance is not None:
            offset = centerx - frame_center
           
            if distance > follow_distance:
                if abs(offset) < align_tolerance:
                    motor1.forward(0.2)
                    motor2.forward(0.2)
                elif offset > 0:
                    motor1.forward(0.4)
                    motor2.backward(0.4)
                else:
                    motor1.backward(0.4)
                    motor2.forward(0.4)
            else:
                stop_motors()
                sleep(0.5)
                open_claw()
                sleep(0.5)
        else:
            state = SEARCH_PERSON

    cv2.putText(frame, f"STATE: {state}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    #cv2.imshow("view", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_bottle.release()
cap_person.release()
cv2.destroyAllWindows()
stop_motors()
