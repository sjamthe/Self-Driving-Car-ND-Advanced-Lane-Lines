from colorthreshold import *
from mark_lanes import process_img
import cv2
import sys
import os
from moviepy.editor import ImageSequenceClip

def print_results(cnt, left, right):
  #if(left.detected and right.detected):
  print(cnt,left.detected,len(left.points),left.fitcoeff[0],left.fitcoeff[1],
          left.fitcoeff[2],left.avgerror,left.intercept,left.intercept0,
          left.cfilter,left.threshold,
          right.detected,len(right.points),right.fitcoeff[0],right.fitcoeff[1],
          right.fitcoeff[2],right.avgerror,right.intercept,right.intercept0,
          right.cfilter,right.threshold,
          left.radius_of_curvature,left.line_base_pos)

"Function to write list of image filenames to a mp4 outfile"
def makemovie(images, outfile):
  clip = ImageSequenceClip(images,fps=30)
  clip.write_videofile(outfile)

if __name__ == '__main__':

  if(len(sys.argv) != 3):
    print("Usage:",sys.argv[0],"<mp4 file> <target dir>")
    exit(1)

  video = sys.argv[1]
  filename, file_extension = os.path.splitext(video)
  videoout = filename+'-out'+ ".mp4" #Output video filename
  directory = sys.argv[2] #to put processed images
  os.mkdir(directory)

  cap = cv2.VideoCapture(video)
  while not cap.isOpened():
      cap = cv2.VideoCapture(video)
      cv2.waitKey(1000)
      print("Wait for the header")

  print(cap.get(cv2.CAP_PROP_FRAME_COUNT),"frames found in",video)

  left = None
  right = None
  images = []
  pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
  while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print(str(pos_frame)+" frames")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #process_image expects RGB
        resultimg, left,right = process_img(frame, left, right)
        #convert image back to NGR
        resultimg = cv2.cvtColor(resultimg, cv2.COLOR_RGB2BGR)
        print_results(pos_frame, left, right)
        #Store results
        outfile = directory + "/frame%d.jpg" % pos_frame
        cv2.imwrite(outfile, resultimg)  # save frame as JPEG file
        images.append(outfile)
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        print("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break
    #if(pos_frame >= 10):
    #  exit()
  cap.release()
  makemovie(images, videoout)
  print("All done!")
