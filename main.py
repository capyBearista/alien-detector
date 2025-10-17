import cv2
import mediapipe as mp # Note: mediapipe only supports Python 3.8-3.11
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

vidcap = cv2.VideoCapture(0)
# Testing with an image
#vidcap = cv2.VideoCapture('Right_Hand_Palm.png')

# Save video
fps = vidcap.get(cv2.CAP_PROP_FPS)
width = int(vidcap.get(3))
height = int(vidcap.get(4))

output = cv2.VideoWriter("vids/1_clip.mp4",
                        cv2.VideoWriter_fourcc('m','p','4','v'),
                        fps=fps, frameSize=(width,height))

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        frame = cv2.flip(frame, 1) # unmirror
        h, w, _ = frame.shape

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Palm landmarks (rough boundary)
                palm_idx = [1, 5, 9, 13, 17, 0]
                palm_points = np.array(
                    [(int(hand_landmarks.landmark[i].x * w),
                      int(hand_landmarks.landmark[i].y * h))
                     for i in palm_idx],
                    np.int32
                )

                # Create a mask for the palm area
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [palm_points], 255)

                # Convert to grayscale for edge detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply Canny edge detection to the palm area only
                edges = cv2.Canny(gray, 60, 150)
                palm_edges = cv2.bitwise_and(edges, edges, mask=mask)

                # Find contours (potential palm lines)
                contours, _ = cv2.findContours(
                    palm_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Draw detected palm line contours
                cv2.drawContours(frame, contours, -1, (0, 255, 255), 1)

                # Draw the palm region outline
                cv2.polylines(frame, [palm_points], True, (255, 225, 225), 2)

                # Print contour coordinates
                for cnt in contours:
                    for point in cnt:
                        x, y = point[0]
                        print(f"Line point: ({x}, {y})")

                # 

        output.write(frame)
        cv2.imshow("Palm Line Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vidcap.release()
cv2.destroyAllWindows()