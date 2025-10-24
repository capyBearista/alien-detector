from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

# Hand landmark drawing settings
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

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

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# Function to calculate the Euclidean distance between two 3D landmarks
def calculate_distance(p1, p2):
    return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

# Define finger connections by their landmark indices
FINGER_LANDMARKS = {
    'Thumb': [1, 2, 3, 4],
    'Index Finger': [5, 6, 7, 8],
    'Middle Finger': [9, 10, 11, 12],
    'Ring Finger': [13, 14, 15, 16],
    'Pinky': [17, 18, 19, 20]
}

def get_finger_lengths(landmarks):
    finger_lengths = {}
    for finger_name, indices in FINGER_LANDMARKS.items():
        length = 0
        # Sum the distance between consecutive joints
        for i in range(len(indices) - 1):
            p1 = landmarks[indices[i]]
            p2 = landmarks[indices[i+1]]
            length += calculate_distance(p1, p2)
        finger_lengths[finger_name] = length
    return finger_lengths
    
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os
from pathlib import Path

aliens_folder = Path("./alien_hands_detected")
# Check if folder already existed
if os.path.exists(aliens_folder):
    for hand in aliens_folder.glob("*"):
        if hand.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            hand.unlink()
            print(f"Deleted: {hand}")
else:
    # If folder does not exist, create one
    os.makedirs(aliens_folder)
    print(f"Created new folder: {aliens_folder}")

# STEP 2: Create an HandLandmarker object.
# Changed to 1 hand instead of 2
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Changed input to web cam
# Open webcam
vidcap = cv2.VideoCapture(0)  

# Save video output
fps = vidcap.get(cv2.CAP_PROP_FPS)
width = int(vidcap.get(3))
height = int(vidcap.get(4))

output = cv2.VideoWriter("vids/1_clip.mp4",
                        cv2.VideoWriter_fourcc('m','p','4','v'),
                        fps=fps, frameSize=(width,height))

paused = False
ALIEN_HAND_DETECTED = 5
alien_hand_in_frame = 0
num_pauses = 0

while vidcap.isOpened():
    if not paused:
        success, frame = vidcap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Convert the frame from BGR (OpenCV) to RGB (Mediapipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # STEP 4: Run hand detection
        detection_result = detector.detect(mp_image)
        idx_fin_len = 0
        mid_fin_len = 0
        diff = 1

        # If a hand is found, check the finger lengths
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            lengths = get_finger_lengths(hand_landmarks)

            idx_fin_len = lengths['Index Finger']
            mid_fin_len = lengths['Middle Finger']
            
            diff = abs(idx_fin_len - mid_fin_len)


        # STEP 5: Draw landmarks on hand and show result. 
        annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
        display_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


        if diff > 0.0001 and alien_hand_in_frame < ALIEN_HAND_DETECTED:
            alien_hand_in_frame += 1
            print(f"Index={idx_fin_len:.4f}, Middle={mid_fin_len:.4f}, Diff={diff:.4f}\n alien_hand_in_frame={alien_hand_in_frame}")
            
        elif diff <= 0.001 and alien_hand_in_frame == ALIEN_HAND_DETECTED:
            print(f"YOU'RE AN ALIEN!!!")
            paused = True
            num_pauses += 1

    else:
        # While paused, output the hand image with landmarks and keep showing the same frame
        # until user presses key 'c'
        # cv2.putText(annotated_image, )
        cv2.imwrite(str(aliens_folder) + "/" + str(num_pauses) + "_alien_hand.jpg", display_frame)

        cv2.putText(display_frame, "ALIEN TRAITS DETECTED - Press 'c' to continue scanning", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    
    output.write(display_frame)
    cv2.imshow("Alien Detector", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and paused:
        paused = False  # continue when 'c' pressed

vidcap.release()
cv2.destroyAllWindows()