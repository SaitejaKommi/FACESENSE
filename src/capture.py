import cv2
import os

os.makedirs("data/raw_frames", exist_ok=True)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imwrite("data/raw_frames/frame.jpg", frame)

cap.release()
print("Saved frame.jpg")
