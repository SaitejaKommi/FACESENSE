import cv2
import numpy as np

prev_frame = None
blink_counter = 0

def detect_blinks(frame):
    global prev_frame, blink_counter

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (7,7), 0)

    if prev_frame is None:
        prev_frame = frame
        return False

    diff = cv2.absdiff(prev_frame, frame)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    movement = np.sum(thresh)

    prev_frame = frame

    # Blink threshold (very light)
    if 200000 < movement < 900000:
        blink_counter += 1

    if blink_counter >= 1:
        blink_counter = 0
        return True

    return False
