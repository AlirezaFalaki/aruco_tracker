import numpy as np
import cv2
from cv2 import aruco
from filess_for_detection.object_detector import HomogeneousBgDetector

cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()


    key = cv2.waitKey(1)
    if key == ord("q"):
        break



cv2.waitKey(0)
cv2.destroyAllWindows()