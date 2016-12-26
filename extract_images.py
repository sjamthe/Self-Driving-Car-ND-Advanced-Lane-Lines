import cv2
import sys
import os

print(cv2.__version__)

if(len(sys.argv) != 3):
  print("Usage:",sys.argv[0],"<mp4 file> <target dir>")
  exit(1)

video = sys.argv[1]
directory = sys.argv[2]
os.mkdir(directory)

cap = cv2.VideoCapture(video)
while not cap.isOpened():
    cap = cv2.VideoCapture(video)
    cv2.waitKey(1000)
    print("Wait for the header")

print(cap.get(cv2.CAP_PROP_FRAME_COUNT),"frames found in",video)
count=0
pos_frame = round(cap.get(cv2.CAP_PROP_POS_FRAMES))
while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        #cv2.imshow('video', frame)
        cv2.imwrite(directory + "/frame%d.jpg" % count, frame)  # save frame as JPEG file
        pos_frame = round(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(str(pos_frame)+" frames")
        count+=1
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

cap.release()
