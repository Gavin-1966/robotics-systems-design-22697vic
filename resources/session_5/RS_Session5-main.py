import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from pi_camera import close_camera, open_camera, read_frame

# Session 5 student scaffold.
# TODO: build the preprocessing pipeline one stage at a time.
# Pipeline= Raw> Gray> Blur> Binary> Clean, ready for decision

camera = open_camera()

kernel = np.ones((5, 5), np.uint8)

while True:
    frame = read_frame(camera)
    if frame is None:
        break

    # TODO: convert to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # TODO: test Gaussian and median blur.
    # blurred = cv2.GaussianBlur(gray, (5,5), 0) # Image smoothing, remove tiny details
    blurred = cv2.medianBlur(gray, 5) # Remove speckles but preserves edges
    
    # TODO: create a binary image.
    # binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY) # Very basic
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # TODO: apply one morphology operation to improve reliability.
    cleaned = cv2.erode(binary, kernel, iterations=1) # Remove small white shapes and white noise
    # cleaned = cv2.dilate(binary, kernel, iterations=1) # Expand whites, fills gaps and connects lines
    # cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # Removes noise and cleans speakles
    # cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)# Fills holes and repairs gaps

    cv2.imshow("session05_student", cleaned)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

close_camera(camera)
cv2.destroyAllWindows()
