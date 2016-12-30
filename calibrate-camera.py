"""
This program is used to calibrate a camera image from lenses distortion.
It uses openCV's findChessboardCorners and calibrateCamera functions.

The input to the program is at least 9 chessboard images taken by that camera
in several different angles (slant positions etc). These images are stored
in camera_cal folder and are named calibration[1..n].jpg
This program assumes the chessboard pattern has ROW_CORNERS=9 and COL_CORNERS=6
change it if you have a different image.

The output from the program is a file named calibration.json that contains
camera Matrix and distortion coefficients returned by cv2.calibrateCamera

As long as the camera focus is kept the same the calibration is needed only one.

For more detail refer:
http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

"""
import numpy as np
import cv2
import glob
import json

#Constants
ROW_CORNERS = 9
COL_CORNERS = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((COL_CORNERS*ROW_CORNERS,3), np.float32)
objp[:,:2] = np.mgrid[0:ROW_CORNERS, 0:COL_CORNERS].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (ROW_CORNERS,COL_CORNERS), None)
    # If found, add object points, image points
    if ret == True:
      print("For image ",fname,"found ",len(corners), "points")
      objpoints.append(objp)
      imgpoints.append(corners)

      # Draw and display the corners
      #cv2.drawChessboardCorners(img, (ROW_CORNERS,COL_CORNERS), corners, ret)

    else:
      print("No points found for image ",fname)

    #cv2.imshow(fname, img)
    #cv2.waitKey(0)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

mtx = mtx.tolist()
dist = dist.tolist()
with open('calibration.json', 'w') as f:
  json.dump([mtx,dist],f)
