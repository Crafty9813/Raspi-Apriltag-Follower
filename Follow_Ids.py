import cv2
import apriltag
import numpy as np
from gpiozero import Motor
import time

# Motor setup
motor1 = Motor(18, 17)
motor2 = Motor(23, 22)
SPEED_FORWARD = 0.6
SPEED_BACKWARD = 0.6

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set lower resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

fx = 600.0
fy = 600.0
cx = 320.0
cy = 240.0
camera_params = (fx, fy, cx, cy)

# Timeout for tag loss
TAG_LOSS_TIMEOUT = 1.5  # in seconds
last_seen_tag_1 = time.time()
last_seen_tag_2 = time.time()
last_seen_tag_3 = time.time()  # New variable for tag 3

turn_start_time = None
turning = False

def follow_april_tag(tag_center, frame_center):
    if tag_center[0] < frame_center[0] - 50:
        motor1.backward(SPEED_BACKWARD)
        motor2.forward(SPEED_FORWARD)
    elif tag_center[0] > frame_center[0] + 50:
        motor1.forward(SPEED_FORWARD)
        motor2.backward(SPEED_BACKWARD)
    else:
        motor1.forward(SPEED_FORWARD)
        motor2.forward(SPEED_FORWARD)

def stop_motors():
    motor1.stop()
    motor2.stop()

def turn_in_place():
    global turning, turn_start_time
    if not turning:
        motor1.forward(0.7)
        motor2.backward(0.8)
        turn_start_time = time.time()
        turning = True
    elif time.time() - turn_start_time > 0.4:  # Turn for 0.4 seconds
        stop_motors()
        turning = False
        time.sleep(1.5)  # Pause before next action

state = "SEARCH_1"
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

    if state == "SEARCH_1":
        if len(detections) > 0 and detections[0].tag_id == 1:
            state = "FOLLOW_1"
        else:
            turn_in_place()

    elif state == "FOLLOW_1":
        if len(detections) > 0 and detections[0].tag_id == 1:
            last_seen_tag_1 = time.time()  # update the timestamp
            tag_center = (int(detections[0].center[0]), int(detections[0].center[1]))
            follow_april_tag(tag_center, frame_center)
        elif time.time() - last_seen_tag_1 > TAG_LOSS_TIMEOUT:
            state = "SEARCH_2"

    elif state == "SEARCH_2":
        if len(detections) > 0 and detections[0].tag_id == 2:
            state = "FOLLOW_2"
        else:
            turn_in_place()

    elif state == "FOLLOW_2":
        if len(detections) > 0 and detections[0].tag_id == 2:
            last_seen_tag_2 = time.time()  # update timestamp
            tag_center = (int(detections[0].center[0]), int(detections[0].center[1]))
            follow_april_tag(tag_center, frame_center)
        elif time.time() - last_seen_tag_2 > TAG_LOSS_TIMEOUT:
            state = "SEARCH_3"

    elif state == "SEARCH_3":
        if len(detections) > 0 and detections[0].tag_id == 3:
            state = "FOLLOW_3"
        else:
            turn_in_place()

    elif state == "FOLLOW_3":
        if len(detections) > 0 and detections[0].tag_id == 3:
            last_seen_tag_3 = time.time()  # update timestamp
            tag_center = (int(detections[0].center[0]), int(detections[0].center[1]))
            follow_april_tag(tag_center, frame_center)
        elif time.time() - last_seen_tag_3 > TAG_LOSS_TIMEOUT:
            stop_motors()
            print("finished")

    else:
        stop_motors()

    # Display detections
    for detection in detections:
        (ptA, ptB, ptC, ptD) = detection.corners.astype(int)
        cv2.line(frame, tuple(ptA), tuple(ptB), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptB), tuple(ptC), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptC), tuple(ptD), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptD), tuple(ptA), (0, 255, 0), 2)

        (cX, cY) = (int(detection.center[0]), int(detection.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stop_motors()
