# -*- coding: utf-8 -*-
__author__ = 'Seran'

import cv2

# 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
vidcap = cv2.VideoCapture('./test_video1.mp4')

count = 0
'''
while vidcap.isOpened():
    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
    ret, image = vidcap.read()

    # 캡쳐된 이미지를 저장하는 함수
    cv2.imwrite("./images/frame%d.jpg" % count, image)

    print('Saved frame%d.jpg' % count)
    count += 1

'''
# get() 함수를 이용하여 전체 프레임 중 1/20의 프레임만 가져와 저장

while vidcap.isOpened():
    ret, image = vidcap.read()
    if int(vidcap.get(1)) % 20 == 0:
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        cv2.imwrite("./images/frame%d.jpg" % count, image)
        img = cv2.imread(r"./images/frame%d.jpg" % count)
        cv2.line(img, (0, 0), (0, img.shape[0]), (255, 0, 0), 1, 1)
        cv2.imwrite("./images/frame%d_grid.jpg" % count, img)

        print('Saved frame%d.jpg' % count)
        count += 1

vidcap.release()
