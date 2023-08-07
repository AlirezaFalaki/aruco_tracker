import numpy as np
import cv2

def bigest_contour(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img,(3,3),0)
    blank = np.zeros(shape=img.shape,dtype='uint8')
    canny = cv2.Canny(img, 20, 255)
    cv2.imshow('canny-FUN', canny)
    contours, hier = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    li = []
    max_area = 0
    for i in range(len(contours)):
        area = (cv2.contourArea(contours[i]))
        if area > max_area :
            max_area = area
            index = i
        tu = [i, area]
        li.append(tu)
    print(li)
    cv2.drawContours(blank,contours,50,255,thickness=1)
    return blank
