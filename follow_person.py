"""
Continuously follow a person until a certain distance away. Its like having a pet dog :)
"""

import cv2
from ultralytics import YOLO
from gpiozero import Motor
from gpiozero import Servo
from time import sleep

model = YOLO('yolov8n_ncnn_model')
person_id = 0 # YOLO class ID for person (from COCO dataset)
person_width = 30 # 30 cm
focal_len = 840
align_tolerance = 100
grab_distance = 55

claw = Servo(12, min_pulse_width=0.0006, max_pulse_width=0.0023)
claw.value = 0
sleep(0.5)
claw.detach()

motor1 = Motor(18, 17)
motor2 = Motor(23, 22)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def stop_motors():
    motor1.stop()
    motor2.stop()

def close_claw():
    claw.value = -0.3
    sleep(0.5)
    claw.detach()

def open_claw():
    claw.value = 0
    sleep(0.5)
    claw.detach()

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret: break
   
    frame_center = frame.shape[1] // 2
    key = cv2.waitKey(1) & 0xFF
    distance = None
    centerx = None
   
    # YOLO detection
    results = model.predict(frame, imgsz=320, verbose=False)[0]
   
    for bb in results.boxes:
        cls_id = int(bb.cls[0])
        if cls_id != person_id:
            continue
       
        xmin, ymin, xmax, ymax = map(int, bb.xyxy[0])
        pixel_width = xmax-xmin
        centerx = (xmin+xmax) // 2
        centery = (ymin+ymax) // 2
       
        if pixel_width > 0:
            distance = (focal_len * person_width)/pixel_width
       
        class_name = model.names[cls_id]
       
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2) # Bounding box
        cv2.circle(frame, (centerx, centery), 5, (0, 0, 255), -1) # Center
       
        label = f"{class_name}, Dist: {distance:.2f} cm"
        cv2.putText(frame, label, (xmin, ymin+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
   
    if distance and centerx:
        offset = centerx-frame_center
        if distance > grab_distance:
            if abs(offset) <= align_tolerance:
                motor1.forward(0.3)
                motor2.forward(0.3)
            elif offset > 0:
                motor1.forward(0.4)
                motor2.backward(0.4)
            elif offset < 0:
                motor1.backward(0.4)
                motor2.forward(0.4)
        else:
            stop_motors()
               
    if key == ord('q'): break
    cv2.imshow('frame', frame)
   
cap.release()
cv2.destroyAllWindows()
stop_motors()
