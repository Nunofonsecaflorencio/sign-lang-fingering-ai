import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os
import numpy as np
import tensorflow as tf
import json
import warnings
import cv2
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated*", category=UserWarning)

hand_model_path = os.path.join('models', 'hand_landmarker.task')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    num_hands=1,
    # min_hand_detection_confidence=0.1,
    # min_hand_presence_confidence=0.1,
    # min_tracking_confidence=0.1,
    # running_mode=VisionRunningMode.VIDEO
    )


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

DATASET_PATH = os.path.join('dataset')

def load_signmodel(file_path=os.path.join('models', 'model.keras')):


  # Load the labels from the JSON file  
  with open('labels.json', 'r') as f:
      actions = json.load(f)
  return tf.keras.models.load_model(file_path), actions

def process_frame(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    
    draw_landmarks_on_image(image, results)
    data = process_hand_landmarks(results)
    return data, image

def draw_landmarks_on_image(image, results):
  mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
  # Draw right hand connections  
  mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


def process_hand_landmarks(results):
    num_landmarks = 21
    num_coordinates = 3

    # Initialize empty arrays with zeros for padding
    left_hand_array = np.zeros(num_landmarks * num_coordinates)
    right_hand_array = np.zeros(num_landmarks * num_coordinates)

    if results.left_hand_landmarks:  
      left_hand_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten()
    
    if results.right_hand_landmarks:
      right_hand_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten()
    
  
    # Combine the arrays
    array = np.concatenate([left_hand_array, right_hand_array])
    min_val = np.min(array)
    max_val = np.max(array)
    
    # Check for division by zero
    if max_val - min_val == 0:
        return np.zeros_like(array)
    
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array