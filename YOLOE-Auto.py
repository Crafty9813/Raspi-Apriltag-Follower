import cv2
from ultralytics import YOLOE
from gpiozero import Motor
from gpiozero import Servo
from time import sleep

model = YOLOE("yoloe-11s-seg.pt")
prompt = ["bottle"]
model.set_classes(prompt)

bottle_width = 6.5
focal_len = 840

claw = Servo(19)
motor1 = Motor(18, 17)
motor2 = Motor(23, 22)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
def stop_motors():
    motor1.stop()
    motor2.stop()
    
def grab_autonomous(center_x, frame_center, distance):
    offset = center_x - frame_center
    align_tolerance = 30
    
    while distance > 17:
        if abs(offset) <= align_tolerance:
            motor1.forward(0.4)
            motor2.forward(0.4)
        elif offset > 0:
            motor1.forward(0.7)
            motor2.backward(0.7)
        elif offset < 0:
            motor1.backward(0.7)
            motor2.forward(0.7)
    stop_motors()
    
key_pressed = False

# MAIN LOOP
while True:
    for i in range(5): cap.grab()
        
    ret, frame = cap.read()
    frame_center = frame.shape[1] // 2
    key = cv2.waitKey(1) & 0xFF
    
    if not ret: break
    
    results = model.predict(frame)
    result = results[0]
    
    for bb in result.boxes:
        xmin, ymin, xmax, ymax = bb.xyxy[0].int().tolist()
        pixel_width = xmax-xmin
        
        if pixel_width > 0:
            distance = (focal_len * bottle_width)/pixel_width
        else:
            distance = 0
            
        centerx = (xmin+xmax) // 2
        centery = (ymin+ymax) // 2
        
        class_name = model.names[int(bb.cls)]
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.circle(frame, (centerx, centery), 7, (0, 0, 255), -1)
        
        label = f"{class_name}, Dist: {distance:.2f} cm"
        cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.imshow('frame', frame)
    
    if key == ord('w'):
        if not key_pressed:
            motor1.forward(0.7)
            motor2.forward(0.7)
            key_pressed = True    
    elif key == ord('a'):
        if not key_pressed:
            motor1.backward(0.8)
            motor2.forward(0.8)
            key_pressed = True
        
    elif key == ord('s'):
        if not key_pressed:
            motor1.backward(0.7)
            motor2.backward(0.7)
            key_pressed = True
            
    elif key == ord('d'):
        if not key_pressed:
            motor1.forward(0.8)
            motor2.backward(0.8)
            key_pressed = True
    
    elif key == ord('p'):
        grab_autonomous(centerx, frame_center, distance)
            
    if key_pressed:
        stop_motors()
        key_pressed = False
    
    if key == ord('q'): break
    
cap.release()
cv2.destroyAllWindows()
