# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()  # 인자값받는 instance
ap.add_argument("-v", "--video", default="./test_video1.mp4", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=4000, help="minimum area size")
args = vars(ap.parse_args())
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:  # 없는 경우니까 넘김~
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])
# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]  # frame[1]로 생각
    text = "Unoccupied"
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)  # 크기 조정
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이색으루
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # 엣지가 남아있는 상태에서 블러링 -> 노이즈 제거 위함
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)  # 두 프레임 사이의 다른 부분 절대값 계산
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]  # 임계값 이미지 return
    # 임계값 적용 함수: threshold(이미지프레임, 임계값, 임계값 넘었을 때 적용값, 바이너리타입:임계값보다 크면 255, 작으면 0)
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)  # 밝은 부분 팽창함수(dilate 두번 겹치게?-> 굵게됨)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE) # (1)thresh에서 (2)바깥 컨투어 (3)라인을 그릴 수 있는 포인트만 반환
    cnts = imutils.grab_contours(cnts) # 컨투어의 총 갯수 값
    # print(cnts)
    # loop over the contours
    for c in cnts: # 컨투어의 갯수값만큼
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]: # 폐곡선(컨투어)안 면적이 min_area보다 작으면 무시
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text

        x, y, w, h = cv2.boundingRect(c) # 컨투어에 외접하는 직사각형 xy좌표, 너비, 높이 ㄴ얻기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # frame, 시작, 끝, 초록색, 선두께2
        text = "Occupied"
        # draw the text and timestamp on the frame # 프레임에 글자쓰는거. 넘겨~
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(50) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release() # 비디오가 없으면 stop 아님 release
cv2.destroyAllWindows()
