from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time
from datetime import datetime

# Function to play a video in a separate window using OpenCV
import threading
import pygame
def play_video(video_path, audio_path=None):
    def _play():
        if audio_path:
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Detection Video", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyWindow("Detection Video")
        if audio_path:
            pygame.mixer.music.stop()
    threading.Thread(target=_play, daemon=True).start()

# Hand landmark drawing settings
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# Classification/overlay settings
ALIEN_THRESHOLD = 0.02
OVERLAY_DURATION_SEC = 2.0
OVERLAY_POS = (30, 50)
COLOR_ALIEN = (0, 0, 255)   # Red (BGR)
COLOR_HUMAN = (0, 255, 0)   # Green (BGR)
COLOR_NO_HAND = (0, 255, 255)  # Yellow (BGR)

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

# STEP 2: Create a HandLandmarker object.
# Changed to 1 hand instead of 2
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Use web cam
# Open webcam
vidcap = cv2.VideoCapture(0)

# Ensure video output directory exists
os.makedirs("vids", exist_ok=True)

# Save video output
fps = vidcap.get(cv2.CAP_PROP_FPS)
try:
    fps_val = float(fps)
    if fps_val <= 0 or np.isnan(fps_val):
        fps_val = 30.0
except Exception:
    fps_val = 30.0

width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if width <= 0 or height <= 0:
    width, height = 640, 480

output = cv2.VideoWriter("vids/1_clip.mp4",
                        cv2.VideoWriter_fourcc('m','p','4','v'),
                        fps_val, (width, height))

# Overlay and freeze state
overlay_text = None
overlay_color = (0, 0, 0)
overlay_until = 0.0  # time.monotonic() when overlay expires
frozen = False
frozen_frame = None
detection_result = None

while vidcap.isOpened():
    if not frozen:
        success, frame = vidcap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Convert the frame from BGR (OpenCV) to RGB (Mediapipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run hand detection every frame so landmarks are drawn live
        detection_result = detector.detect(mp_image)

        # Draw landmarks on hand and show result.
        annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
        display_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Render transient overlay if active (non-frozen)
        if overlay_text and time.monotonic() < overlay_until:
            cv2.putText(display_frame, overlay_text, OVERLAY_POS,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, overlay_color, 2)
    else:
        # When frozen, show the frozen frame with persistent overlay
        if frozen_frame is None:
            # Recover by unfreezing if frame missing
            frozen = False
            continue
        display_frame = frozen_frame.copy()
        if overlay_text:
            cv2.putText(display_frame, overlay_text, OVERLAY_POS,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, overlay_color, 2)

    # Write and display frame
    output.write(display_frame)
    cv2.imshow("Alien Detector", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        if not frozen:
            # Analyze current frame's landmarks once, then freeze
            if detection_result and detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                lengths = get_finger_lengths(hand_landmarks)
                idx_len = lengths['Index Finger']
                mid_len = lengths['Middle Finger']
                diff = abs(idx_len - mid_len)

                if diff <= ALIEN_THRESHOLD:
                    label = "ALIEN"
                    color = COLOR_ALIEN
                    # Save annotated still
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    save_path = aliens_folder / f"{timestamp}_alien_hand.jpg"
                    cv2.imwrite(str(save_path), display_frame)
                    play_video("alien.mp4", "alien.mp3")  # Play alien video with sound
                else:
                    label = "HUMAN"
                    color = COLOR_HUMAN
                    play_video("human.mp4", "human.mp3")  # Play human video with sound

                print(f"Captured (frozen): Index={idx_len:.6f}, Middle={mid_len:.6f}, Diff={diff:.6f} -> {label}")
                overlay_text = f"{label} DETECTED (diff={diff:.6f})"
                overlay_color = color
            else:
                # No hand detected at capture time
                print("Capture pressed but no hand detected")
                overlay_text = "No hand detected - press 'c' to resume"
                overlay_color = COLOR_NO_HAND

            # Freeze on capture
            frozen_frame = display_frame.copy()
            frozen = True
        else:
            # Resume live detection
            frozen = False
            overlay_text = None
            overlay_until = 0.0

vidcap.release()
output.release()
cv2.destroyAllWindows()