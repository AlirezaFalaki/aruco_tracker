import cv2
from filess_for_detection.object_detector import *
import matplotlib.pyplot as plt
import numpy as np
from aruco_detector import  Marker



# Load live camera

cap = cv2.VideoCapture('https://192.168.194.131:8080/video')
#cap = cv2.VideoCapture('https://22.93.230.169:8080/video')
#cap = cv2.VideoCapture('https://22.88.126.199:8080/video')
#cap = cv2.VideoCapture('192.168.42.2')
#cap = cv2.VideoCapture(0)


# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
first_frame = cv2.imread('image_aruco/img.jpg')
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
canny = cv2.Canny(blurred, 70, 215)

i=0
while True:
    _,img = cap.read()
    img = cv2.resize(img,(1200,800),interpolation=cv2.INTER_AREA)
    cv2.imshow('org',img)
    marker = Marker(img,canny)
    dst = marker.crop_picture()
    img_marker = marker.detect_marker(dst)
    center_marker = marker.marker_location()
    #print(center_marker,print(type(center_marker)))
    li = []
    if center_marker != [] :
        if center_marker.ndim == 1 :
            li.append(marker.calculate(center_marker[0], center_marker[1], 120, 84, [0, 0], [1200, 0], [0, 800]))
        else:
            for center in center_marker:
                 li.append(marker.calculate(center[0],center[1],120,84,[0,0],[1200,0],[0,800]))
    print(li)
    # if center_marker != [] :
    #     for center in center_marker:
    #         li.append(marker.calculate(center[0],center[1],120,84,[0,0],[1200,0],[0,800]))
    # print(li)

    if i==10:
        cv2.imwrite('image_aruco/img.jpg',img)
    i += 1

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


    # if center:cdcd
    #     print(marker.calculate(center[0],center[1],150,100,listpoint['top_left'] ,listpoint['top_right'] ,\
    #                 listpoint['buttom_left']))
    #output = pose_estimation(img,cv2.aruco.DICT_7X7_50, intrinsic_camera, distortion)

    #cv2.imshow("Image", img)
    # key = cv2.waitKey(1)
    # if key == ord("q"):
    #     break



