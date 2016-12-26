import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import sys
import os

def visualize(img, warped):
    # Visualize undistortion, warped
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(warped)
    ax2.set_title('Undistorted,warped Image', fontsize=30)
    plt.show()

""" For lanes case src points should have same y axis for top & bottom points"""
def adjust_src(image, src):
    maxHeight = image.shape[0]*0.92 #90% is arbitrary to exclude bumper image
    print(src)
    src[1][1] = src[0][1]
    src[2][1] = src[3][1]
    #extrapolate the trapezoid so it goes to the bottom of the image
    #point 0,3 are left side, 1,2 are right side
    x1 = src[0][0]
    y1 = src[0][1]
    x2 = src[3][0]
    y2 = src[3][1]
    lefty = maxHeight
    leftx = x1 - np.abs((x1-x2)/(y1-y2)*(y1-lefty))
    # replace the point values
    src[3][0] = leftx
    src[3][1] = lefty

    x1 = src[1][0]
    y1 = src[1][1]
    x2 = src[2][0]
    y2 = src[2][1]
    righty = maxHeight
    rightx = x1 + np.abs((x1-x2)/(y1-y2)*(y1-righty))
    # replace the point values
    src[2][0] = rightx
    src[2][1] = righty
    print(src)
    return src

def four_point_transform(image, src):

    src = adjust_src(image, src)
    #use image size
    maxWidth = image.shape[0]
    maxHeight = image.shape[1]
    #maxWidth = np.round(src[2][0]-src[3][0])
    #maxHeight = np.round(src[2][1]-src[1][1])

    dst = np.array([
        [0., 0.],
        [maxWidth, 0.],
        [maxWidth, maxHeight],
        [0., maxHeight]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped, M

def read_calibrations():
    with open('calibration.json', 'r') as f:
        [mtx,dist] = json.load(f)
        mtx = np.array(mtx)
        dist = np.array(dist)

    return mtx, dist

if __name__ == '__main__':

    if(len(sys.argv) < 2):
      print("Usage: ",sys.argv[0],"<imagefile>")
      exit(1)

    fname = sys.argv[1]
    if(os.path.isfile(fname) is False):
      print("Usage: ",sys.argv[0],"<valid imagefile>")
      exit(1)

    mtx, dist = read_calibrations()
    img = plt.imread(fname)
    dstimg = cv2.undistort(img, mtx, dist, None, mtx)

    #Capture points from screen
    plt.imshow(img)
    points = plt.ginput(4,timeout=60) #captures two clicks x,y coordinates in points array
    plt.close()
    print(points)
    points = np.array(points, dtype = "float32")
    warped, M = four_point_transform(dstimg, points)
    # Visualize undistortion, warped
    visualize(img,warped)

    M = M.tolist()
    with open('transform.json', 'w') as f:
        json.dump(M,f)

    #let us try second images based on matrix M
    with open('transform.json', 'r') as f:
        M1 = json.load(f)
        M1 = np.array(M1)
        fname = sys.argv[2]
        img = plt.imread(fname)
        dstimg = cv2.undistort(img, mtx, dist, None, mtx)
        warped = cv2.warpPerspective(dstimg, M1, (dstimg.shape[0], dstimg.shape[1]))
        visualize(img,warped)
