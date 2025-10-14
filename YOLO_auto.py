import cv2
from ultralytics import YOLO
from gpiozero import Motor
from gpiozero import Servo
from time import sleep

model = YOLO('yolov8n.pt')
bottle_id = 39

bottle_width = 6.5
focal_len = 840

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
   
'''
def grab_autonomous(center_x, frame_center, distance):
    offset = center_x - frame_center
    align_tolerance = 30
   
    while distance > 55:
        if abs(offset) <= align_tolerance:
            motor1.forward(0.3)
            motor2.forward(0.3)
        elif offset > 0:
            motor1.forward(0.7)
            motor2.backward(0.7)
        elif offset < 0:
            motor1.backward(0.7)
            motor2.forward(0.7)
    stop_motors()'''
   
key_pressed = False

# MAIN LOOP
while True:
    ret, frame = cap.read()
    frame_center = frame.shape[1] // 2
    key = cv2.waitKey(1) & 0xFF
   
    if not ret: break
   
    results = model.predict(frame, conf=0.4, verbose=False)[0]
   
    for bb in results.boxes:
        cls_id = int(bb.cls[0])
       
        if cls_id != bottle_id:
            continue
       
        xmin, ymin, xmax, ymax = map(int, bb.xyxy[0])
        pixel_width = xmax-xmin
       
        if pixel_width > 0:
            distance = (focal_len * bottle_width)/pixel_width
           
        centerx = (xmin+xmax) // 2
        centery = (ymin+ymax) // 2
       
        class_name = model.model.names[cls_id]
       
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.circle(frame, (centerx, centery), 5, (0, 0, 255), -1)
       
        label = f"{class_name}, Dist: {distance:.2f} cm"
        cv2.putText(frame, label, (xmin, ymin+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       
    cv2.imshow('frame', frame)
   
    if key == ord('w'):
        if not key_pressed:
            motor1.forward(0.5)
            motor2.forward(0.5)
            key_pressed = True    
    elif key == ord('a'):
        if not key_pressed:
            motor1.backward(0.8)
            motor2.forward(0.8)
            key_pressed = True
       
    elif key == ord('s'):
        if not key_pressed:
            motor1.backward(0.5)
            motor2.backward(0.5)
            key_pressed = True
           
    elif key == ord('d'):
        if not key_pressed:
            motor1.forward(0.8)
            motor2.backward(0.8)
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
   
    if key == ord('q'): break
   
cap.release()
cv2.destroyAllWindows()
stop_motors()
