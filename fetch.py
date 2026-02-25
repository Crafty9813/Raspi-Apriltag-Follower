"""
Robot should follow a ball and give it back to the person like playing fetch.
Sadly there is some trouble getting it to recognize the ball for some reason :(
"""

import cv2
from ultralytics import YOLO
from gpiozero import Motor, Servo
from time import sleep

model = YOLO("yolov8n_ncnn_model", task="detect")

# YOLO class IDs
PERSON_ID = 0
BALL_ID = 32

person_width = 30 # cm
ball_width = 6.5 # cm
focal_len = 840

align_tolerance = 80
grab_distance = 15 # cm
follow_distance = 60

# STATES
FOLLOW_BALL = 0
SEARCH_PERSON = 1
FOLLOW_PERSON = 2
SEARCH_BALL=3

state = FOLLOW_BALL

# HARDWARE
motor1 = Motor(18, 17)
motor2 = Motor(23, 22)

claw = Servo(12, min_pulse_width=0.0006, max_pulse_width=0.0023)
claw.value = 0
sleep(0.5)
claw.detach()

cap_ball = cv2.VideoCapture(0)
cap_person = cv2.VideoCapture(2)

for cap in (cap_ball, cap_person):
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

# MAIN LOOP
while True:
    ret_ball, frame_ball = cap_ball.read()
    ret_person, frame_person = cap_person.read()

    if not ret_ball or not ret_person:
        print("Camera read failed")
        break

    # Choose camera based on state
    if state == FOLLOW_BALL or state == SEARCH_BALL:
        frame = frame_ball
        target_id = BALL_ID
        target_width = ball_width
    else:
        frame = frame_person
        target_id = PERSON_ID
        target_width = person_width

    frame_center = frame.shape[1] // 2
    detected = False
    distance = None
    centerx = None

    # YOLO
    results = model.predict(frame, imgsz=320, conf=0.15, verbose=False)[0] # Decreasing confidence doesn't help much still

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

    # FSM
    if state == FOLLOW_BALL:
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
                state = SEARCH_BALL
        else:
            state = SEARCH_PERSON
           
    elif state == SEARCH_BALL:
        motor1.forward(0.4)
        motor2.backward(0.4)

        if detected and target_id == BALL_ID:
            stop_motors()
            state = FOLLOW_BALL

    # Show current state
    cv2.putText(frame, f"STATE: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Robot view", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_ball.release()
cap_person.release()
cv2.destroyAllWindows()
stop_motors()
