# Self-Driving-Car-ND-Advanced-Lane-Lines

This is project 4 of SDCND. It uses CV with different techniques (unwarping,thresholding etc) to detect lanes

## Step 1 - Calibrate Camera for lens distortion
calibrate-camera.py program is used to calibrate a camera image from lenses distortion. The input to the program is at least 9 chessboard images taken by that camera in several different angles (slant positions etc).
The output from the program is a file named calibration.json that contains camera Matrix and distortion coefficients returned by cv2.calibrateCamera.

## Step 2 - Create warp matrix to see top view of the road from 3D image
warped.py program helps us to change a 3D perspective to a 2D view so we can see the road lanes in parallel instead of merging in the horizon. It uses cv2.getPerspectiveTransform function to get a 3x3 matrix that can be used to warp the image from 3D to 2D.

You need to run this program only once for a setting of the camera on the mount. Once the camera is mounted take pictures that clearly show a grid with vertical and horizontal parallel lines.
Note: When we warp the image we actually switch the width and height as we want to see a long image (lanes)
Image will be 720x1280

The program needs calibration.json to remove camera distortion. It produces transform.json as output that contains the transform matrix.

# Show output of warped.py here

## Pipeline to process a video and mark lanes
process_video.py is the main driver program that implements a pipeline to process images
in a video. It reads the mp4 video file from argv[1], calls mark_lanes.process_img function
for each image, writes the image in a folder named by argv[2].
In the end it writes converts all the output images into a video named argv[1]-out.mp4

# Link output of the two youtube videos here.

## Marking Lanes
Program mark_lanes.py has all the logic of identifying the lanes, marking them and calculating the curvature of the road and distance of center of car (camera) from lane center.

### Process_img function
This function is the main entry point, it takes front image as an input along with prev lanes.
1. It calibrates the image for camera distortion using calibration.json
2. It transforms the image using transform.json so we get top view of the lanes.
3. It calls extract_lanes function that has all the logic of converting image to lanes.
4. It calls final_lanes function that averages the lane from last 5 consecutive lanes. If a lane is not identified then we simply use prev lane as the new lane. If prev lane is None as well then we use extrapolate left lane from the right one and vice versa based on previous lane width.
5. It calls drawlanes to draws lane marking on the image and writes on image
6. It returns image with lane markings

### extract_lanes function
1. Splits the image in the middle for left lane and right lane
2. calls best_fit_line function for each image.
3. returns the best lanes for this image

### best_fit_line function
Given an input image find the best points that can make a lane
1. Apply region_of_interest
2. calls lane_points with various color filters (yellow and white)
3. call comparelines to compare lines from all filters and select best one
4. compare lane for each color filter with last n images
5. returns best lane

### Color thresholds
After experimenting with various color filters and thresholds like HLS, HSV etc
and looking at S or V channels I decided to create color thresholds that enhance yellow
and white lanes.

#### Yellow threshold
Yello threshold basically ignores the blue color.
In an RGB image is blue intensity is > 110 we make is 0 then we grayscale the image and apply threshold.

#### White threshold
A white color means RGB all three intensities are same or close. So instead of gray scaling the image
we apply threshold only to points that have all three color intensities in the threshold range if not we
set intensity to 0.

### lane_points function
Given an image find the best points that can make a line
1. input image is color warped, apply threshold to get binary image
   we apply a range of thresholds lower thresholds for yellowfilter range from 110-200
   and for white filter range from 50-200
2. split the image into horizontal stripes of num_of_stripes (32)
3. create a histogram for each stripe. Each stripe is 1280/32 = 40 points high
4. We find the x axis location of points that have maximum histogram for each strip
5. points from all the strips are fed to clean_points function
6. polyfit a quadratic curve on he points passed by clean_points
7. Call comparelines to compare all lines found by this algo and select the best one
8. Make sure line selected by comparelines is a valid line.

### cleanpoints function
This function removes points that don't predict the lane lines
1. We split image into vertical stripes that are 100 points wide on x axis.
2. Calculate histogram for each stripe and select stripes with maxhist
3. We may have multiple stripes with maxhist as lane markings are broad,
  find how far apart they are. We permit a spread of 20 points, anything more
  may be shadow etc.
4. The goal of this function is to only keep points that are atmost 20 points apart

### isLineValid function
Compare the line with previous image line and return true if line is valid.
1. This function compares the top and bottom x intercept of new line with prevline
 if intercept is more than 13 we reject this line as car cannot shift so much between
 consecutive images, so this line may not be valid.
2. The fitcoeff[1] of this line is like a slope, we want to make sure it is > 0.75 for valid lines
3. We apply leftlimit and rightlimit for the lines as the lanes cannot go far away from the image

### compareLines function
comparelines compares current line with prev line
if the line is valid line compared with prevline then it compares it with
last bestline and returns the one that is best.
It returns None is line is not valid and last bestline is None
Logic: This functions assumes that any good line should have at least 4 unique points
to describe the line. When we compare 2 lines the one with more unique points (upto 10)
is preferred. If lines have > 10 unique points then we prefer line that has less spread
i.e its points are closer to each other.

