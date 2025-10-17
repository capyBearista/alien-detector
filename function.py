from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

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

# STEP 2: Create an HandLandmarker object.
# Changed to 1 hand instead of 2
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("hand3.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# TEST: PRINT OUT THE LENGTH OF ALL FINGERS
if detection_result.hand_landmarks:
        # Assuming only one hand is detected
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Calculate and print finger lengths
        lengths = get_finger_lengths(hand_landmarks)
        index_finger_length = 0.0
        middle_finger_length = 0.0
        for finger, length in lengths.items():
            # print(f'{finger} Length: {length:.4f} (units are relative)')
            if finger == 'Index Finger':
                index_finger_length = length
            elif finger == 'Middle Finger':
                middle_finger_length = length
        print(f'Index Finger Length: {index_finger_length:.4f} (units are relative)')
        print(f'Middle Finger Length: {middle_finger_length:.4f} (units are relative)')
        distance_between_index_and_mid_finger = max(index_finger_length,middle_finger_length) - min(index_finger_length,middle_finger_length)
        if distance_between_index_and_mid_finger < 0.1:
            print(f'The difference between two fingers: {distance_between_index_and_mid_finger:.4f}')
            print("YOU'RE AN ALIEN!!!")

        # TEST IF INDEX AND MIDDLE FINGER ARE ABOUT THE SAME LENGTH


# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


cv2.waitKey(0)
cv2.destroyAllWindows()