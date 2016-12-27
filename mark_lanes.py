import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import sys
import os
import colorthreshold as ct
import random
from line import Line
import math
from copy import deepcopy

""" Debug function for visualizing lanes """
def visualize(img,img2,img3,fname):
    # Visualize undistortion, warped
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title(fname, fontsize=20)
    ax2.imshow(img2,cmap='gray')
    ax3.imshow(img3,cmap='gray')

    return ax2,ax3


def polyfit(img, line):

  ymax = img.shape[0]
  yvals = line.points[:,1]
  if(line.isLeft):
    xmax = img.shape[1]
    xmin = 0
    xvals = line.points[:,0]
  else:
    #I this is right lane adjust the x points
    xmax = img.shape[1]*2
    xmin = img.shape[1]
    xvals = line.points[:,0] + img.shape[1]
    #line.points = line.points + [img.shape[1], 0]

  fit,residuals, _, _, _ = np.polyfit(yvals, xvals, 2, full=True)
  #print("left=",line.isLeft,"fit=",fit,"residual = ",residuals)
  line.fitcoeff = fit
  if(len(residuals) > 0):
    line.avgerror = math.sqrt(residuals[0])/len(xvals)
  else:
    line.avgerror = sys.maxsize

  #find points on the curve if we need to draw the lanes
  #image is too tall, start from a bit late
  top = 600
  ys = np.linspace(top,ymax,5) #choose equidistant points
  fitx = fit[0]*ys**2 + fit[1]*ys + fit[2]
  fitx = fitx
  #intercept where the lane meets the bottom of the image
  line.intercept = int(fitx[-1])
  #intercept where the lane meets the top of the image
  line.intercept0 = int(fitx[0])
  #print("x intercepts = ",line.intercept, line.interept0)

  points = []
  cnt=0
  for (x, y) in zip(fitx, ys):
    x = int(x)
    if(x >= xmin and x<= xmax):
      points.append([x,y])

  line.fitpoints = np.array(points)
  return line

""" Draws points and best fit line for visual debug only
"""
def drawpoints(ax, line):
  if(line is None or line.points.size == 0):
    return

  xvals = line.points[:,0]
  yvals = line.points[:,1]
  if(np.mean(xvals) <= 360): #left lane
    ax.plot(xvals, yvals, 'o', color='red')
  else:
    ax.plot(xvals, yvals, 'o', color='blue')
  if(len(line.fitpoints) > 0):
    ax.plot(line.fitpoints[:,0], line.fitpoints[:,1], color='green', linewidth=1)

def region_of_interest(img):
  # zero out points outside region of interest
  # We ignore points to left of leftlimit and to the right of rightlimit

  leftlimit = 50
  rightlimit = img.shape[1]-50

  img[:,0:leftlimit] = 0
  img[:,rightlimit:img.shape[1]] = 0

  toplimit = 0
  img[0:toplimit] = 0

  return img


""" This function removes points that don't predict the lane lines
    We split image into vertical stripes of ver_stripes
"""
def clean_points(points, line, width):
  newpoints = []
  #print ("points.shape = ", points.shape)
  xvals = points[:,0]
  yvals = points[:,1]
  stripe_width = 100 # 30 points wide strip
  #vert_stripes = int((max(xvals) - min(xvals))/stripe_width) + 1
  vert_stripes = int(width/stripe_width)
  #print(vert_stripes, max(xvals), min(xvals))
  histogram = np.histogram(xvals,vert_stripes)
  #print (line.isLeft, histogram)
  maxhist = max(histogram[0])
  #print("vertical maxhist",maxhist)
  if(maxhist > 2): #we need atleast 3 points
    #Find out the point of maxhist so we can get the range
    i = np.where(histogram[0] == maxhist)[0]
    #print(maxhist,i,histogram)
    minx = histogram[1][i[0]]
    maxx = histogram[1][i[-1]+1]
    spread = maxx-minx
    #print("spread = ",spread,minx,maxx)
    # We want to include all points that are within 50 points of each other
    SPREAD_WIDTH_LIMIT = 20 #Make this less if we get too many bad points
    if(spread <= SPREAD_WIDTH_LIMIT):
      maxspread = histogram[1][vert_stripes]-histogram[1][0]
      if(maxspread <= SPREAD_WIDTH_LIMIT):
        #include all points as they are too close
        minx = histogram[1][0]
        maxx = histogram[1][vert_stripes-1]
      else:
        #We need to widen our selection, decide how wide
        leftpoints = -1
        rightpoints = -1
        if(i[0] > 0):
          leftpoints = histogram[0][i[0]-1]
        if(i[-1] < vert_stripes-1):
          rightpoints = histogram[0][i[-1]+1]
        if(spread*2>=50):
          #take one nearest one left or right whichever has more points
          if(leftpoints >= rightpoints):
            minx = histogram[1][i[0]-1]
          else:
            maxx = histogram[1][i[-1]+1]
        else:
          # we can increase spread by two bands preferably left & right
          if(i[0] > 0):
            minx = histogram[1][i[0]-1]
          if(i[-1] < vert_stripes-1):
            maxx = histogram[1][i[-1]+1]
          if(i[0] == 0):
            maxx = histogram[1][i[-1]+2] #min cannot move so move max by 2
          if(i[-1] == vert_stripes):
            minx = histogram[1][i[0]-2] #max cannot move so move min by 2

    #print("max, min",maxx,minx)
    #print("shape = ",xvals.shape,minx,maxx)
    for cnt in range(xvals.shape[0]):
      if(xvals[cnt] >= minx and xvals[cnt] <= maxx):
        newpoints.append(points[cnt])
      #else:
        #print("skipping ",cnt,points[cnt])
    return newpoints
  else:
    #print("ASSERT: No points found after clean_points, points shape=",points.shape)
    return (points)

""" given an image find the best points that can make a lane
    1. input image is color warped, apply threshold to get binary image
    2. split the image into horizontal stripes of num_of_stripes
"""
def lane_points(warped, prevlane, line):

  num_of_stripes = 32 #split image height in these many sections
  height = warped.shape[0]
  width = warped.shape[1]
  cfilter = line.cfilter

  bestline = None

  for threshold in range(200,50,-10):
    if(cfilter == 'S'):
      img = ct.S_threshold(warped,threshold)
    elif(cfilter == 'yellow'):
      img = ct.yellow_threshold(warped,threshold)
    elif(cfilter == 'white'):
      img = ct.white_threshold(warped,threshold)
    else:
      img = ct.gray_threshold(warped,threshold)

    line.threshold = threshold
    points = []
    step = int(height/num_of_stripes)

    for y in range(0,height,step):
      #Find the sum of points on each vertical line in the strip
      histogram = np.sum(img[y:y+step,0:width], axis=0)
      #print(y, histogram)
      #find the max of the histogram to find tallest line
      maxhist = max(histogram)
      #print(y, maxhist)
      if(maxhist > 1): #There can be maximum (1280/32)=40 points in a stripe
        #find out the x location of each vertical line with max points
        xvals = np.where(histogram == maxhist)[0]
        #print("xvals=",len(xvals),xvals)
        #if we have too many x columns with maxhist skip as we may have shadow etc
        if(len(xvals) < 20): #Was 50
          #for each xval we add a point in the middle of the step height
          for x in xvals:
            #print("point=",[x,y+step/2])
            points.append([x,y+step/2])
        #else: #Debug
        #  print("skipping as xvals >= 20",lane,y,len(xvals),cfilter,threshold)
      #else: #Debug
      #  print("skipping as maxhist is",maxhist,"for",y,cfilter,threshold)
    #end of for range
    if(len(points) > 2): #We should atleast have 3 steps with points
      #clean points will draw vertical stripes and pick the best points
      points = clean_points(np.array(points),line, width)
      line.points = np.array(points)

      line = polyfit(img, line)
      bestline = comparelines(prevlane, bestline, line)
      #print("received ",cfilter, threshold,bestline.points.shape[0],line.points.shape[0])

      if(0 and bestline is not None): #debug
        print("found left?",line.isLeft,cfilter,threshold)
        ax2,ax3 = visualize(warped,img,img, "in lane_points "+str(line.isLeft))
        pts = np.array(points)
        ax2.plot(pts[:,0],pts[:,1], 'o', color='red')
        ax2.set_title(cfilter + " " + str(threshold) )
        bestpts = np.array(bestline.points)
        ax3.plot(bestpts[:,0],bestpts[:,1], 'o', color='blue')
        ax3.set_title("best "+ cfilter + " " + str(bestline.threshold) )
        plt.show()

  #end of for theshold
  if(0 and bestline is not None): #debug
    print("best line for",line.isLeft, cfilter,bestline.threshold)

  return bestline

"""
  Compare the line with previous best line and return true if line is valid
"""
def islinevalid(prevlane, line):
  shiftAllowed = 10
  leftintLimit = 200
  # If the xintercept and xintercept0 of line are within shiftAllowed from prevLane
  # return True
  if(prevlane is not None):
    bottomshift = abs(line.intercept - prevlane.intercept)
    topshift = abs(line.intercept0 - prevlane.intercept0)
    #print("shift",line.isLeft,bottomshift,topshift)
    if((bottomshift > shiftAllowed) or
      (topshift > shiftAllowed)):
      return False

  #print("coeff = ",line.isLeft, line.cfilter, line.fitcoeff[1])
  if(abs(line.fitcoeff[1]) > 0.75):
    return False

  #Adding a check to avoid 1st image of challenge mistaken by white filter
  if(line.isLeft and line.intercept >= leftintLimit):
    return False

  return True

""" comparelines compares current line with prev line
    if the line is valid line compared with prevline then it compares it with
    last bestline and returns the one that is best.
    It returns None is line is not valid and last bestline is None
"""
def comparelines(prevlane, bestline, line):

  #if line is valid an nothing to compare against we return the line
  # else none
  if(bestline is None):
    if islinevalid(prevlane, line):
      #print("returning back line",line.isLeft,line.cfilter,line.threshold )
      return deepcopy(line) #We have nothing to compare against
    else:
      #print ("returning None")
      return None

  # If we are here then we have bestline. Still if line is not valid return
  # bestline , else compare this line with bestline
  if islinevalid(prevlane, line) is False:
    #print("returning bestline",bestline.isLeft,bestline.cfilter,bestline.threshold )
    return bestline

  lineunique = np.unique(line.points[:,1]).shape[0]
  linexspread = max(line.points[:,0]) - min(line.points[:,0])
  bestunique = np.unique(bestline.points[:,1]).shape[0]
  bestxspread = max(bestline.points[:,0]) - min(bestline.points[:,0])

  if(0):
    print("Comparing bestline",line.isLeft, bestline.cfilter,bestline.threshold, bestunique,bestxspread,'with',
            line.cfilter,line.threshold, lineunique,linexspread)
  limit = 10
  # if we have more than limit lineunique take line with less xspread
  if((lineunique < 15 and lineunique > bestunique) or
     (lineunique >= 15 and linexspread < bestxspread )):

    if(0): #debug
      print("selected new bestline:",line.isLeft,line.cfilter,line.threshold ,lineunique,linexspread,
        "old bestline",bestline.isLeft,bestline.cfilter,bestline.threshold ,bestunique,bestxspread)
    prevbestline = bestline
    prevbestunique = bestunique
    bestline = deepcopy(line)
    bestunique = lineunique
    bestxspread = linexspread

  #else:
   # print("returning bestline2",bestline.isLeft,bestline.cfilter,bestline.threshold,
   #       "as", bestunique,bestxspread, "better than",line.cfilter,line.threshold, lineunique,linexspread)
  return bestline

""" Given an input image find the best points that can make a lane
    1. calls lane_points with various color filters
    2. from points returned from lane_points calculate best_fit_poly
    3. compare lane for each color filter with last n images
    4. returns best lane
"""
def best_fit_line(warped, prevlane, isLeft):
  warped = region_of_interest(warped)

  #colorfilters = ['yellow','white']
  colorfilters = ['white','yellow']
  #colorfilters = ['yellow']
  bestline = None
  images = [warped,warped]
  i=0
  for cfilter in colorfilters:
    line = Line()
    line.isLeft = isLeft
    line.cfilter = cfilter
    line = lane_points(images[i], prevlane, line)
    i=i+1
    if(line is None):
      continue #try another cfilter

    if(0): #debug
      ax2,ax3 = visualize(warped,warped,warped, "in best_fit_line "+str(line.isLeft))
      ax2.set_title(line.cfilter +"="+str(line.threshold))
      drawpoints(ax2,line)
      if(bestline is not None):
        ax3.set_title(bestline.cfilter +"="+str(bestline.threshold))
      drawpoints(ax3,bestline)
      plt.show()

    #compare this line with previous one and last bestline
    bestline = comparelines(prevlane, bestline, line)

  #if(bestline is None):
  #  prevlane.detected = False
  #  print(prevlane.isLeft, "returning prev lane as bestline is None")
  #  return prevlane #that is the best we have.
  #else:
  #  bestline.detected = True
  return bestline

""" Input is a warped image we need to return the best left & right lanes
    1. split image into left & right
    2. call lane_points for each image to get best left and right points
    3. call polyfit to fit the points to lanes
    4. compare new fit wit historic lines and return best lanes
"""
def extract_lanes(warped, prevleft, prevright):

  #split the image vertically in left & right
  midpoint = int(warped.shape[1]/2)
  leftimg = warped[:,0:midpoint]
  #leftimg = region_of_interest(leftimg)
  leftline = best_fit_line(leftimg, prevleft, True)

  rightimg = warped[:,midpoint:warped.shape[1]]
  #rightimg = region_of_interest(rightimg)
  rightline = best_fit_line(rightimg, prevright, False)

  if(0): #debug
    ax = visualize(warped,warped,"both")
    drawpoints(ax,leftline)
    drawpoints(ax,rightline)
    plt.show()

  return leftline, rightline

""" Draw the lanes and fill the color and returns the image with lanes marked
"""
def drawlanes(warped, leftline, rightline):
  # Create an image to draw the lines on
  warp_zero = np.zeros_like(warped).astype(np.uint8)

  #we cannot draw if both lanes are not there
  if(leftline.fitpoints is None or rightline.fitpoints is None):
    print("ASSERT: left or right lane missing, cannot draw")
    return warp_zero
  # Recast the x and y points into usable format for cv2.fillPoly()
  #flipud changes the order of right points from top to bottom.
  #don't know why we need flipud yet.
  pts_left = np.array([leftline.fitpoints])
  pts_right = np.array([np.flipud(rightline.fitpoints)])
  try:
    pts = np.hstack((pts_left, pts_right))
    pts = np.array(pts,'int32')

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_zero, pts, (0,255, 0))
  except (ValueError):
    print("ValueError: pts_left",pts_left)
    print("ValueError: pts_right",pts_right)
  return warp_zero

mtx = None
dist = None
M = None

""" averagelane : take average of last N lines
"""
def averagelane(line, prevlane):
  N = 3

  #assert(line.isLeft == prevlane.isLeft), "lanes should have same isLeft"
  if(prevlane is not None and
      len(prevlane.recent_xpoints) >= N):
    #print("N = ",N)
    prevlane.recent_xpoints.pop(0) #remove oldest point only
  #add new points
  if(line is not None and prevlane is not None):
    xpoints = line.fitpoints[:,0] #this is an array
    xpoints = xpoints.tolist() #make this a list.
    #add to the old list.
    prevlane.recent_xpoints.append(xpoints)

    line.recent_xpoints = prevlane.recent_xpoints #copy for next time
    #calculate the new fit xpoints as average of these
    #print("before",prevlane.fitpoints)
    fitx = np.mean(prevlane.recent_xpoints,axis=0)
    line.fitpoints[:,0] = fitx
    #intercept where the lane meets the bottom of the image
    #line.intercept = int(fitx[-1])
    #intercept where the lane meets the top of the image
    #line.intercept0 = int(fitx[0])
    #print("after",prevlane.fitpoints)
    yvals = prevlane.fitpoints[:,1]
    line.fitcoeff,_, _, _, _ = np.polyfit(yvals, fitx, 2, full=True)
    #print("left=",line.isLeft,"fit=",fit,"residual = ",residuals)
    line.detected = True
  elif(prevlane is None and line is not None and line.fitpoints.shape[0] > 0):
    #print("shape",line.fitpoints.shape)
    xpoints = line.fitpoints[:,0] #this is an array
    xpoints = xpoints.tolist() #make this a list.
    line.recent_xpoints = [xpoints]
    line.detected = True
  else:
    line = prevlane
    if(line is not None):
      line.detected = False
  return line

""" extrapolateleft subtracts lanewidth from right line points
"""
def extrapolateleft(rightline):
  LANEWIDTH = 313

  left = deepcopy(rightline)
  left.detected = False
  left.recent_xpoints = [] #initialize
  left.fitpoints[:,0] = left.fitpoints[:,0] - LANEWIDTH

  return left

""" extrapolateright adds lanewidth to leftlane points
"""
def extrapolateright(leftline):
  LANEWIDTH = 280

  right = deepcopy(leftline)
  right.detected = False
  right.recent_xpoints = [] #initialize
  right.fitpoints[:,0] = right.fitpoints[:,0] + LANEWIDTH

  return right

""" final_lanes take is historic perspective. It averages best n lanes if a
  line was found. If it is not found it takes from prevlane is prev is not None
  if prevlane is None is decided lane based on parallel lane available
"""
def final_lanes(left, right, prevleft, prevright):

  if((prevleft is not None) or
      (left is not None and prevleft is None)): #if left is none we assign prevlane here
    left = averagelane(left,prevleft)

  if((prevright is not None) or
      (right is not None and prevright is None)): #if right is None we assign prevright here
    right = averagelane(right,prevright)

  if(left is None and prevleft is None and right is not None):
    left = extrapolateleft (right)
    left.detected = False
    print("Leftlane extrapolated")
  if(right is None and prevright is None and left is not None):
    right = extrapolateright (left)
    right.detected = False
    print("rightlane extrapolated")

  if(left is None):
    print("ASSERT: left is None")
  if(right is None):
    print("ASSERT: right is None")

  return left, right

"""Wrapper to do all steps on an image, returns image with lanes
    1. takes image, prev leftlane and rightlane as input
    2. (read calibrations if None)
    3. Warp the image
    4. call extract_lanes to get left & right lanes
    5. draw lanes on the image
    6. unwarp the image with lanes marked
    7. Return image, left, right lane object
"""
def process_img(originalimg, prevleft=None, prevright=None):
  global mtx, dist,M #added so we read mtx only once

  if(mtx is None or dist is None or M is None): #read these only once for efficiency
    mtx, dist, M = ct.read_calibrations()

  warped = ct.warp(originalimg, mtx, dist, M) #warp the image to top view
  left, right = extract_lanes(warped, prevleft, prevright) #extract lines from warped img
  left, right = final_lanes(left, right, prevleft, prevright)

  color_warp = drawlanes(warped, left, right) #draw lanes on img
  resultimg = ct.unwarp(color_warp, M, originalimg) #unwarp to original view

  return resultimg, left, right

""" Main function for testing purpose. Input sequence of images """
if __name__ == '__main__':

  if(len(sys.argv) < 2):
    print("Usage:",sys.argv[0],"<image1> <image2> ...")
    exit(1)

  left = None
  right = None
  for fname in sys.argv[1:]:
    print (fname)
    originalimg = plt.imread(fname) #read image
    resultimg,left,right = process_img(originalimg, left, right)
    plt.figure(figsize=(20,10))
    plt.title(fname, fontsize=20)
    plt.imshow(resultimg) #debug display
    print(left.isLeft,left.cfilter,left.threshold)
    print(right.isLeft,right.cfilter,right.threshold)
    plt.show()
