import numpy as np
from scipy import signal
import cv2

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
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # Shi-Tomasi의 코너점
# print(p0)  # shape : (코너점 갯수, 1, 2)[[[a,b]], ..., [[c,d]]]

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
count = 1
while True:
    ret, frame = cap.read()
    # print(frame.shape) # 높이, 너비, 채널의 수
    if ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # param : 이전 프레임, 추적할 이전 포인트, 다음 프레임 등을 인자로 전달
        # return : 이전 프레임에서 추적할 포인트가 연속된 다음 프레임에서 추적될 경우 상태값 1을 반환, 아님 0
        #          추적할 포인트가 이동한 새로운 위치값도 함께 반환
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)  # 이전 프레임, 이전 점, 다음 프레임

        # Select good points
        good_new = p1[st == 1] # 다음 프레임에서 추척할 포인트가 이동한 위치값들의 배열.
        # print(good_new.shape) # (점갯수, 2) <- 2가 x값, y값인듯
        good_old = p0[st == 1] # 다음 프레임에서 추적되는 포인트의 위치값들의 배열
        p0 = p1 # 다음 프레임이 이전 프레임으로
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 새로 p0이 된 거 그레이색상으로 바꾸기
        vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR) # 했던거는 다시 rgb색상으로 바꾸기

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)): # 예전, 지금 추적 포인트 배열 순회
            a, b = old.ravel() # x, y축 뽑기
            c, d = new.ravel() # x, y축 뽑기 (2, )
            # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            # frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            # create line endpoints
            lines = np.vstack([a, b, c, d]).T.reshape(-1, 2, 2) # shape 맞추기.. 어렵돠
            lines = np.int32(lines)
            # create image and draw
            # vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            for (x1, y1), (x2, y2) in lines: # x1 : old, x2 : new point
                # norm : 벡터의 길이 혹은 크기를 측정하는 방법(함수), =원점에서 벡터 좌표까지의 거리
                if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) < 2:
                    continue # 이전 포인트와 다음 포인트의 차의 벡터 크기가 2보다 작으면 pass
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2) # 아님 line긋고
                cv2.circle(vis, (x2, y2), 1, (0, 0, 255), 2) # 콩나물 대가리(다음 포인트에)
        # img = cv2.add(frame,mask)
        # cv2.imwrite("./images/frame%d.jpg" % count, img)
        count = count + 1 # 아마 필요없는듯?
        cv2.imshow('frame', vis)

    k = cv2.waitKey(50) & 0xff
    if k == 27:
        break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy() # 이전꺼 변수에 넣고
        p0 = good_new.reshape(-1, 1, 2) # shape 이쁘게

cv2.destroyAllWindows()
cap.release()
