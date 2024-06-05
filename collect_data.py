import uuid
import os
import cv2
import numpy as np
import time
import mediapipe as mp
from preprocessing import *


dataset_path = os.path.join('dataset')
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    
actions = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
print("Select an action from the list or create a new one:")
for i, action in enumerate(actions, 1):
    print(f"{i}. {action}")
print(f"{len(actions) + 1}. Create a new action")

choice = int(input("Enter the number of your choice: "))
if choice == len(actions) + 1:
    action = input("Enter the name of the new action: ")
    action_path = os.path.join(dataset_path, action)
    os.makedirs(action_path)
else:
    action = actions[choice - 1]
    action_path = os.path.join(dataset_path, action)
    

with HandLandmarker.create_from_options(options) as landmarker:
    # Start a webcam capture session:
    cap = cv2.VideoCapture(0)
    timestamp = frame_count =  0
    start_time = time.time()
    TIMER = 5 # seconds
    
    while True:
        timestamp += 1
        # Capture a frame from the webcam:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        dt = current_time - start_time
        
        # Annotate the image with the timer
        cv2.putText(frame, f"Recording: {action.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"Captured: {frame_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"Countdown: {int(TIMER - dt + 1)}s", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        data, frame = process_frame(frame, landmarker, timestamp) 
        if dt >= TIMER:
            start_time = current_time
            if data.max() > 0:
                frame_count += 1
                np.save(os.path.join(dataset_path, action, str(uuid.uuid4())), data)


        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Display the frame with hand landmarks:
        cv2.imshow('Hand Landmarker', frame)

        # Press 'q' to quit:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam capture:
    cap.release()
    cv2.destroyAllWindows()
