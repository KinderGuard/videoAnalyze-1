import numpy as np
from scipy import signal
import cv2

cap = cv2.VideoCapture('./test_video1.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
count=1
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    p0 = p1
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = old.ravel()
        c,d = new.ravel()
        #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        # create line endpoints
        lines = np.vstack([a,b,c,d]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        # create image and draw
        #vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        for (x1, y1), (x2, y2) in lines:
            if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) < 2:
                continue
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(vis, (x2, y2), 1, (0, 0, 255), 2)
    #img = cv2.add(frame,mask)
    #cv2.imwrite("./images/frame%d.jpg" % count, img)
    count=count+1
    cv2.imshow('frame',vis)
    k = cv2.waitKey(50) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()