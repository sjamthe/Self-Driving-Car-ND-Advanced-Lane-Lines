"""
This is a class definition for Line class used by mark_lanes.py

"""
import numpy as np
import sys

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xpoints = []
        #average points of the fitted line over the last n iterations
        #self.bestpoints = None
        #polynomial coefficients averaged over the last n iterations
        #self.best_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float')

        #calculated lane width
        self.lane_width = None
        #array of [x,y] values for detected line pixels
        self.points = []

        #polynomial coefficients for the most recent fit
        self.fitcoeff = [np.array([False])]
        #sqrt of residual returned from polyfit divided by num of points. lower is better
        self.avgerror = sys.maxsize
        #array of [x,y] values derived from current fit line, used for plotting only
        self.fitpoints = None
        #Is left line true to determine left or right line
        self.isLeft = None
        #intercept where the lane meets the bottom of the image
        self.intercept = None
        #intercept where the lane meets the top of the image
        self.intercept0 = None
        #channel/filter that was used for best fit
        self.cfilter = None
        #threshold used for best fit
        self.threshold = None
        #position of frame
        self.frame_pos = -1
