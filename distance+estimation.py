import cv2 as cv
from cv2 import aruco
import numpy as np

calib_data = np.load('MultiMatrix.npz')
#print(calib_data.files)
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
#
MARKER_SIZE = 14
marker_dict = aruco.getPredefinedDictionary(cv.aruco.DICT_7X7_50)
param_markers = aruco.DetectorParameters()

#cap = cv.VideoCapture("https://192.168.18.218:8080/video") #give the server id shown in IP webcam App
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()


            # calculate the distance
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )

            # for pose of the marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 20, 4)
            cv.putText(
                frame,
                #f"id: {ids[0]} Dist: {round(distance, 2)}",
                f"Distance: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            # print(ids, "  ", corners)
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()

#--------------------------------------------------------------------------
#
# import numpy
# import cv2
# from cv2 import aruco
# import os
# import pickle
#
# # Check for camera calibration data
# if not os.path.exists('./calibration.pckl'):
#     print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
#     exit()
# else:
#     f = open('calibration.pckl', 'rb')
#     (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
#     f.close()
#     if cameraMatrix is None or distCoeffs is None:
#         print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
#         exit()
#
# # Constant parameters used in Aruco methods
# ARUCO_PARAMETERS = aruco.DetectorParameters()
# #ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000) original
# ARUCO_DICT = aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
#
# # Create grid board object we're using in our stream
# board = aruco.GridBoard_create(
#         markersX=2,
#         markersY=2,
#         markerLength=0.09,
#         markerSeparation=0.01,
#         dictionary=ARUCO_DICT)
#
# # Create vectors we'll be using for rotations and translations for postures
# rvecs, tvecs = None, None
#
# cam = cv2.VideoCapture(0)
#
# while(cam.isOpened()):
#     # Capturing each frame of our video stream
#     ret, QueryImg = cam.read()
#     if ret == True:
#         # grayscale image
#         gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
#
#         # Detect Aruco markers
#         corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
#
#         # Refine detected markers
#         # Eliminates markers not part of our board, adds missing markers to the board
#         corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
#                 image = gray,
#                 board = board,
#                 detectedCorners = corners,
#                 detectedIds = ids,
#                 rejectedCorners = rejectedImgPoints,
#                 cameraMatrix = cameraMatrix,
#                 distCoeffs = distCoeffs)
#
#
#         QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))
#
#     if ids is not None:
#         try:
#             rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(corners, 10.5, cameraMatrix, distCoeffs)
#             QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 5)
#         except:
#             print("Deu merda segue o baile")
#
#
#         cv2.imshow('QueryImage', QueryImg)
#
#     # Exit at the end of the video on the 'q' keypress
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()