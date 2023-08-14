import numpy as np
import cv2
from cv2 import aruco
#from filess_for_detection.object_detector import HomogeneousBgDetector

img = cv2.imread('ff.jpg', 1)
#img = cv2.imread('image_aruco/img.jpg')

def edge_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(blurred, 70, 215)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(canny, kernel)
    cv2.imshow('canny',dilated)
    return dilated

#cv2.imshow('img1',img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,img = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
#cv2.imshow('img',img)
# تار کردن اولین فریم به روش گاوس
blurred = cv2.GaussianBlur(img, (5, 5), 0)
# تشخیص لبه های عکس بوسله تابع canny
canny = cv2.Canny(blurred, 70, 215)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dilated = cv2.dilate(canny, kernel)
# kernel = np.ones((3, 3), np.uint8)
# canny = cv2.dilate(canny,kernel)
#cv2.imshow('canny', dilated)

contours ,hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

blank = np.zeros(shape=img.shape[:2],dtype='uint8')

c = max(contours,key=cv2.contourArea)
#x,y,w,h = cv2.boundingRect(c)
#print(x,y,w,h)
# draw the biggest contour (c) in green
#cv2.rectangle(blank,(x,y),(x+w,y+h),(0,255,0),2)
cv2.drawContours(blank,c,-1,255,1)


#cv2.imshow('blank',blank)

cv2.waitKey(0)
cv2.destroyAllWindows()