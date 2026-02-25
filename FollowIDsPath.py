"""
Follows AprilTags based on IDs. FSM for different ID following so it can perform a series of follows.
"""

import cv2
import apriltag
import numpy as np
from gpiozero import Motor
import time

motor1 = Motor(18, 17)
motor2 = Motor(23, 22)
SPEED_FORWARD = 0.6
SPEED_BACKWARD = 0.6

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

fx = 600.0
fy = 600.0
cx = 320.0
cy = 240.0
camera_params = (fx, fy, cx, cy)

TAG_DELAY = 0.7
last_seen_tag_1 = time.time()
last_seen_tag_2 = time.time()
last_seen_tag_3 = time.time()

turn_start_time = None
turn_pause_time = None
turning = False
paused = False

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

def spin():
    global turning, turn_start_time, turn_pause_time, paused

    if paused:
        if time.time() - turn_pause_time > 0.5:
            paused = False
        else:
            stop_motors()
            return

    if not turning:
        motor1.forward(0.7)
        motor2.backward(0.8)
        turn_start_time = time.time()
        turning = True
    elif time.time() - turn_start_time > 0.4:
        stop_motors()
        turning = False
        paused = True
        turn_pause_time = time.time()

#Main loop
state = "FIND_1"
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cant capture img")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

    if state == "FIND_1":
        print("FINDING id1")
        if len(detections) > 0 and detections[0].tag_id == 1:
            state = "FOLLOW_1"
        else:
            spin()

    elif state == "FOLLOW_1":
        print("Following id1")
        if len(detections) > 0 and detections[0].tag_id == 1:
            last_seen_tag_1 = time.time()
            tag_center = (int(detections[0].center[0]), int(detections[0].center[1]))
            follow_april_tag(tag_center, frame_center)
        else:
            if time.time() - last_seen_tag_1 > TAG_DELAY:
                state = "FIND_2"
            else:
                stop_motors()

    elif state == "FIND_2":
        print("FINDING id2")
        if len(detections) > 0 and detections[0].tag_id == 2:
            state = "FOLLOW_2"
        else:
            spin()

    elif state == "FOLLOW_2":
        print("Following id2")
        if len(detections) > 0 and detections[0].tag_id == 2:
            last_seen_tag_2 = time.time()
            tag_center = (int(detections[0].center[0]), int(detections[0].center[1]))
            follow_april_tag(tag_center, frame_center)
        else:
            if time.time() - last_seen_tag_2 > TAG_DELAY:
                state = "FIND_3"
            else:
                stop_motors()

    elif state == "FIND_3":
        print("Finding id3")
        if len(detections) > 0 and detections[0].tag_id == 3:
            state = "FOLLOW_3"
        else:
            spin()

    elif state == "FOLLOW_3":
        print("Following id3")
        if len(detections) > 0 and detections[0].tag_id == 3:
            last_seen_tag_3 = time.time()
            tag_center = (int(detections[0].center[0]), int(detections[0].center[1]))
            follow_april_tag(tag_center, frame_center)
        else:
            if time.time() - last_seen_tag_3 > TAG_DELAY:
                stop_motors()
                print("Done! :)")
            else:
                stop_motors()

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
