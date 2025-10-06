import cv2
import numpy as np
from gpiozero import Motor
import time

motor1 = Motor(18, 17)
motor2 = Motor(23, 22)
SPEED = 1

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

def stop_motors():
    motor1.stop()
    motor2.stop()

key_pressed = False

#Main loop
while True:
    ret, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF
    
    if not ret:
        print("Cant capture")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

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

    if key == ord('w'):
        if not key_pressed:
            motor1.forward(SPEED)
            motor2.forward(SPEED)
            key_pressed = True    
    elif key == ord('a'):
        if not key_pressed:
            motor1.backward(SPEED)
            motor2.forward(SPEED)
            key_pressed = True
        
    elif key == ord('s'):
        if not key_pressed:
            motor1.backward(SPEED)
            motor2.backward(SPEED)
            key_pressed = True
            
    elif key == ord('d'):
        if not key_pressed:
            motor1.forward(SPEED)
            motor2.backward(SPEED)
            key_pressed = True
            
    elif key_pressed:
        stop_motors()
        key_pressed = False
        
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stop_motors()
