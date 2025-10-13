import cv2
import numpy as np
from gpiozero import Motor
from gpiozero import Servo
import time

motor1 = Motor(18, 17)
motor2 = Motor(23, 22)
claw = Servo(12, min_pulse_width=0.0006, max_pulse_width=0.0023)

claw.value = 0
sleep(0.5)
claw.detach()

SPEED = 1

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

    elif key == ord('p'):
        if not key_pressed:
            claw.value = -0.3
            sleep(0.5)
            claw.detach()
            key_pressed = True
           
    elif key == ord('o'):
        if not key_pressed:
            claw.value = 0
            sleep(0.5)
            claw.detach()
            key_pressed = True
            
    elif key_pressed:
        stop_motors()
        key_pressed = False
        
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stop_motors()
