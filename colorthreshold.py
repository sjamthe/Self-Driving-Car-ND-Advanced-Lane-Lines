import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import sys
import os

def visualize(img, warped, gray,fname):
    # Visualize undistortion, warped
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title(fname, fontsize=20)
    ax2.imshow(warped,cmap='gray')
    ax2.set_title('Undistorted,warped S_channel', fontsize=20)
    #histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)
    #ax2.plot(histogram)
    ax3.imshow(gray,cmap='gray')
    ax3.set_title('Undistorted,warped gray', fontsize=20)
    points = plt.ginput(1,timeout=20) #captures two clicks x,y coordinates in points array

def read_calibrations():

    with open('calibration.json', 'r') as f:
        [mtx,dist] = json.load(f)
        mtx = np.array(mtx)
        dist = np.array(dist)
    #let us try second images based on matrix M
    with open('transform.json', 'r') as f:
        M = json.load(f)
        M = np.array(M)
    return mtx, dist, M

""" unistort and warp the image, swap the image height and width as we want to
    look at tall image to look at lane lines properly """
def warp(img, mtx, dist, M):
    #print("input = ",img.shape)
    dstimg = cv2.undistort(img, mtx, dist, None, mtx)
    #print("dstimg = ",dstimg.shape)
    #we are swapping height & width to make image look longer
    warped = cv2.warpPerspective(dstimg, M, (dstimg.shape[0], dstimg.shape[1]))
    #print("warped = ",warped.shape)
    return warped

""" Warp the blank back to original image space using inverse perspective matrix (Minv)
    notice warped had swapped height and width to make image look taller than wide
    so we swap the height & width back to normal
"""
def unwarp(color_warp, M, img):
  retval, Minv = cv2.invert(M)
  if(retval > 0):
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    unwarped = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return unwarped

def normalize(rawpoints):
    #Linear stretch from lowest value = 0 to highest value = 100
    high = 255
    low = 0

    mins = np.min(rawpoints)
    maxs = np.max(rawpoints)
    rng = maxs - mins
    print(mins,maxs)

    scaled_points = high - (((high - low) * (maxs - rawpoints)) / rng)
    return scaled_points.astype(int)

def S_threshold(warped, limit=80):
    hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]
    gray = s_channel
    thresh = (limit, 255)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    return binary

def V_threshold(warped, limit=80):
    hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    gray = v_channel
    thresh = (limit, 255)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    return binary

def yellow_threshold(warped, limit=180):
    #input image is RGB
    r_channel = warped[:,:,0]
    g_channel = warped[:,:,1]
    b_channel = warped[:,:,2]

    yellow = np.array(warped, copy=True)
    #subtract the blue color
    yellow[b_channel > 110] = 0 #80-120 worked
    #plt.imshow(yellow)
    #plt.title('yellow '+str(limit), fontsize=20)
    #plt.show()

    gray = cv2.cvtColor(yellow, cv2.COLOR_RGB2GRAY)
    thresh = (limit, 255)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    return binary

def white_threshold(warped, limit=180):
    #input image is RGB
    r_channel = warped[:,:,0]
    g_channel = warped[:,:,1]
    b_channel = warped[:,:,2]

    thresh = (limit, 255)
    binary = np.zeros_like(r_channel)
    #each channel should have threshold above limit for white.
    binary[(r_channel > thresh[0]) & (r_channel <= thresh[1])
        &  (b_channel > thresh[0]) & (b_channel <= thresh[1])
        &  (b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1

    return binary

def gray_threshold(warped, limit=200):
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    thresh = (limit, 255)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

    return binary

if __name__ == '__main__':

    mtx, dist, M = read_calibrations()

    for fname in sys.argv[1:]:
        print (fname)
        img = plt.imread(fname)
        warped = warp(img,mtx, dist, M)
        binary = color_threshold(warped)
        binary1 = color_threshold1(warped)
        visualize(img,binary,binary1,fname)
        cv2.imwrite('binary.jpg',binary)
        cv2.imwrite('binary1.jpg',binary1)
        break
