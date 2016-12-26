import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path

if(len(sys.argv) != 2):
  print("Usage: ",sys.argv[0],"<imagefile>")
  exit(1)

fname = sys.argv[1]
if(os.path.isfile(fname) is False):
  print("Usage: ",sys.argv[0],"<valid imagefile>")
  exit(1)

with open('calibration.json', 'r') as f:
  [mtx,dist] = json.load(f)
  mtx = np.array(mtx)
  dist = np.array(dist)

img = cv2.imread(fname)
dst = cv2.undistort(img, mtx, dist, None, mtx)

#write the file
#cv2.imwrite('Undistorted.jpg', dst)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()
