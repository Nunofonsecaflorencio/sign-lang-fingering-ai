import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os
import numpy as np
import tensorflow as tf
import json
import warnings
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
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.3,
    running_mode=VisionRunningMode.VIDEO)

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

def process_frame(frame, landmarker, timestamp):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    hand_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result)
    data = process_hand_landmarks(hand_landmarker_result)
    return data, annotated_image

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image


def process_hand_landmarks(hand_landmarker_result):
    num_landmarks = 21
    num_coordinates = 3

    # Initialize empty arrays with zeros for padding
    left_hand_array = np.zeros((num_landmarks, num_coordinates))
    right_hand_array = np.zeros((num_landmarks, num_coordinates))

    # Identify hands and extract landmarks
    for i, category in enumerate(hand_landmarker_result.handedness):
        hand_type = category[0].category_name
        landmarks = hand_landmarker_result.hand_landmarks[i]
        landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])

        if hand_type == 'Left':
            left_hand_array = landmarks_array
        elif hand_type == 'Right':
            right_hand_array = landmarks_array

    # Combine and flatten the arrays
    array = np.concatenate([left_hand_array, right_hand_array]).flatten()
    min_val = np.min(array)
    max_val = np.max(array)
    
    # Check for division by zero
    if max_val - min_val == 0:
        return np.zeros_like(array)
    
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array
