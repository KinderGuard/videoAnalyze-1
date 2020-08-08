import numpy as np
from scipy import signal
import cv2
import imutils
import datetime
from PIL import Image
import math

cap = cv2.VideoCapture('./test_video1.mp4')

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.1,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_frame = imutils.resize(old_frame, width=800)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # Shi-Tomasi의 코너점
# print(p0)  # shape : (코너점 갯수, 1, 2)[[[a,b]], ..., [[c,d]]]

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
firstFrame = None
count = 1
px = -1  # 한 프레임 안에서 두 포인트 잇기 위한 변수, 이전 포인트값 저장
py = -1
while True:
    ret, frame = cap.read()
    blank_image = None  # 빈 이미지
    # print(frame.shape) # 높이, 너비, 채널의 수
    if ret:
        frame = imutils.resize(frame, width=800)  # frame 크기 재조정
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, weight, channel = frame.shape
        blank_image = np.zeros((height, weight, 3), np.uint8)  # frame과 크기가 같은 빈 이미지 만들기

        # param : 이전 프레임, 추적할 이전 포인트, 다음 프레임 등을 인자로 전달
        # return : 이전 프레임에서 추적할 포인트가 연속된 다음 프레임에서 추적될 경우 상태값 1을 반환, 아님 0
        #          추적할 포인트가 이동한 새로운 위치값도 함께 반환
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)  # 이전 프레임, 이전 점, 다음 프레임

        # Select good points
        good_new = p1[st == 1]  # 다음 프레임에서 추척할 포인트가 이동한 위치값들의 배열.
        # print(good_new.shape) # (점갯수, 2) <- 2가 x값, y값인듯
        good_old = p0[st == 1]  # 다음 프레임에서 추적되는 포인트의 위치값들의 배열
        p0 = p1  # 다음 프레임이 이전 프레임으로
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 새로 p0이 된 거 그레이색상으로 바꾸기
        vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)  # 했던거는 다시 rgb색상으로 바꾸기
        # px = -1 #이전 포인트와 다음 포인트 잇기 위한 변수, 이전 포인트값 저장
        # py = -1
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):  # 예전, 지금 추적 포인트 배열 순회
            a, b = old.ravel()  # x, y축 뽑기
            c, d = new.ravel()  # x, y축 뽑기 (2, )

            # create line endpoints
            lines = np.vstack([a, b, c, d]).T.reshape(-1, 2, 2)  # shape 맞추기.. 어렵돠
            lines = np.int32(lines)

            # create image and draw
            # vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            for (x1, y1), (x2, y2) in lines:  # x1 : old, x2 : new point

                # norm : 벡터의 길이 혹은 크기를 측정하는 방법(함수), =원점에서 벡터 좌표까지의 거리
                if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) < 2:
                    continue  # 이전 포인트와 다음 포인트의 차의 벡터 크기가 2보다 작으면 pass
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 아님 line긋고
                cv2.circle(vis, (x2, y2), 1, (0, 0, 255), 2)  # 콩나물 대가리(다음 포인트에)
                cv2.circle(blank_image, (x2, y2), 1, (0, 0, 255), 50)  # 빈 이미지에 optical flow 포인트 표시
                if np.linalg.norm(np.array([px, py]) - np.array([x2, y2])) > 6000:  # 이전 포인트와 거리가 멀면 continue
                    px = x2  # 현재 포인트를 이전 포인트로 저장
                    py = y2
                    continue
                #if px >= 0 and py >= 0:  # 이전 포인트와 충분히 가깝고, 첫 프레임이 아닐 경우
                 #   cv2.line(blank_image, (px, py), (x2, y2), (0, 0, 255), 10)
                px = x2  # 현재 포인트를 이전 포인트로 저장
                py = y2
        blank_image = Image.fromarray(blank_image, 'RGB')  # 포인트를 그린 빈 이미지를 rgb형태 이미지 파일 변환
        blank_image.save("./images3/frame%d.png" % count)  # 이미지 저장 *images3 디렉토리 만들기
        # compute the absolute difference between the current frame and
        # first frame
        img = cv2.imread("./images/frame%d.png" % count)  # optical flow 포인트만 그려진 이미지 불러오기
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이색으루
        gray = cv2.GaussianBlur(gray, (21, 21), 0)  # 엣지가 남아있는 상태에서 블러링 -> 노이즈 제거 위함
        thresh = None
        if firstFrame is None:
            firstFrame = gray
            continue
        frameDelta = cv2.absdiff(firstFrame, gray)  # 두 프레임 사이의 다른 부분 절대값 계산
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]  # 임계값 이미지 return
        # 임계값 적용 함수: threshold(이미지프레임, 임계값, 임계값 넘었을 때 적용값, 바이너리타입:임계값보다 크면 255, 작으면 0)
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)  # 밝은 부분 팽창함수(dilate 두번 겹치게?-> 굵게됨)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)  # (1)thresh에서 (2)바깥 컨투어 (3)라인을 그릴 수 있는 포인트만 반환
        cnts = imutils.grab_contours(cnts)  # 컨투어의 총 갯수 값
        print(cnts)
        # loop over the contours
        for c in cnts:  # 컨투어의 갯수값만큼
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 4000:  # 폐곡선(컨투어)안 면적이 min_area보다 작으면 무시
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text

            x, y, w, h = cv2.boundingRect(c)  # 컨투어에 외접하는 직사각형 xy좌표, 너비, 높이 얻기
            # 그림은 원본 프레임에 그리기
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)  # frame, 시작, 끝, 초록색, 선두께2
            text = "Occupied"
            # draw the text and timestamp on the frame # 프레임에 글자쓰는거. 넘겨~
            cv2.putText(vis, "Room Status: {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(vis, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", vis)
        cv2.imshow("Points", img)
        count += 1

    k = cv2.waitKey(50) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()  # 이전꺼 변수에 넣고
    p0 = good_new.reshape(-1, 1, 2)  # shape 이쁘게

cv2.destroyAllWindows()
cap.release()
