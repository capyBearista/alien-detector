import cv2

vidcap = cv2.VideoCapture(0)

if not vidcap.isOpened():
    print("[error] Could not open VideoCapture. Exiting...")
    exit()

# loop forever
while True:
    ret, frame = vidcap.read()

    if not ret:
        print("[warn] Could not open frame.")
        break

    cv2.imshow("Live Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    vidcap.release()
    cv2.destroyAllWindows()