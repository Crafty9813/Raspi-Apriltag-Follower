"""
Using the webcam's mic, robot waits for someone to say any sentence containing "grab" and "water" so for example "Grab me water", then finds and grabs a bottle and delivers it to a person. Then uses a USB speaker to talk to the person: "Hello Jonathan, here is your water."
"""

import cv2
from ultralytics import YOLO
from gpiozero import Motor, Servo
from time import sleep
import subprocess
import json
import pyaudio
from vosk import Model, KaldiRecognizer

MIC_INDEX = 2 # Using webcam's built in mic
RATE = 16000

# YOLO class ids
PERSON_ID = 0
BOTTLE_ID = 39

person_width = 30 # cm
bottle_width = 6.5 # cm
focal_len = 840

align_tolerance = 80
grab_distance = 27 # cm
follow_distance = 55

# STATES
FOLLOW_BOTTLE = 0
SEARCH_PERSON = 1
FOLLOW_PERSON = 2

state = None

spoken = False # So doesn't speak forever

# Motors and hardware
motor1 = Motor(18, 17)
motor2 = Motor(23, 22)

claw = Servo(12, min_pulse_width=0.0006, max_pulse_width=0.0023)
claw.value = 0
sleep(0.5)
claw.detach()

cap_bottle = cv2.VideoCapture(0)
cap_person = cv2.VideoCapture(2)

for cap in (cap_bottle, cap_person):
    cap.set(3, 640)
    cap.set(4, 480)

# Action helper funcs
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
   
yolo_model = YOLO("yolov8n_ncnn_model", task="detect")
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
print("Listening for commands...")

while True:
  	ret_bottle, frame_bottle = cap_bottle.read()
  	ret_person, frame_person = cap_person.read()

	  if not ret_bottle or not ret_person:
	  		print("Camera read failed :(")
	    	break
	  
	  # Camera selection based on state
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
  
	  results = yolo_model.predict(frame, imgsz=320, conf=0.25, verbose=False)[0]
	     
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
		    break # Track one object only
  
	  data = stream.read(800, exception_on_overflow=False)
	  if rec.AcceptWaveform(data):
			  result = json.loads(rec.Result())
			  text = result.get("text", "").lower()
			  if "grab" in text and "water" in text:
					  print("Voice cmd: FOLLOW_BOTTLE")
					  state = FOLLOW_BOTTLE
	  
	  if state == FOLLOW_BOTTLE:
	  		if detected and distance is not None:
	      		offset = centerx - frame_center
	    
				    if distance > grab_distance:
				      	if abs(offset) < align_tolerance:
						        motor1.forward(0.25)
						        motor2.forward(0.25)
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
		    motor1.forward(0.7)
		    motor2.backward(0.7)
	  
			  if detected and target_id == PERSON_ID:
					  stop_motors()
					  state = FOLLOW_PERSON
	  
	  elif state == FOLLOW_PERSON:
			  if detected and target_id == PERSON_ID and distance is not None:
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
								sleep(0.5)
								open_claw()
								sleep(0.5)
								if not spoken:
										subprocess.run([
										"espeak",
										"-a", "200",
										"Hello Jonathan, here is your water."
										])
										spoken = True
										sleep(1)
										motor1.backward(0.3)
										motor2.backward(0.3)
										sleep(3)
			  else:
			  		state = SEARCH_PERSON

cap_bottle.release()
cap_person.release()
cv2.destroyAllWindows()
stop_motors()
