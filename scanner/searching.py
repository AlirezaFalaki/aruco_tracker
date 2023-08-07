import cv2
import numpy as np
import mapper

empty = np.zeros(shape=(1300,800),dtype='uint8')

def crop_picture(self,img):

    image = cv2.resize(img, (1300, 800))
    cv2.imshow('img',image) # resizing because opencv does not work well with bigger images
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB To Gray Scale
    # (5,5) is the kernel size and 0 is sigma that determines the amount of blur
    blurred = cv2.GaussianBlur(gray, (3,3),0)
    edged = cv2.Canny(blurred, 80,210)  # 30 MinThreshold and 50 is the MaxThreshold
    # retrieve the contours as a list, with simple apprximation model
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # the loop extracts the boundary contours of the pagel

    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)

        if len(approx) == 4:
            target = approx
            break
    approx = mapper.mapp(target)  # find endpoints of the sheet

    pts = np.float32([[0, 0], [1200,0], [1200, 800], [0,800]])  # map to 800*800 target window

    op = cv2.getPerspectiveTransform(approx,pts)  # get the top or bird eye view effect
    dst = cv2.warpPerspective(orig, op, (1200, 1200))
    cv2.imshow("Scanned", dst)

# press q or Esc to close
cv2.waitKey(0)
cv2.destroyAllWindows()




