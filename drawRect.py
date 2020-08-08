import cv2
import numpy as np

cap = cv2.VideoCapture('./test_video2.avi')

def draw(rects,color):
 for r in rects:
  p1 = (r[0], r[1])
  p2 = (r[0]+r[2], r[1]+r[3])
  cv2.rectangle(img, p1, p2, color,4)

limit_area = 500
x = 0
y = 0
w = 0
h = 0
nuclei = []
count = 0
number_name = 1

while True:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.add(img, 0.70)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (90, 90, 0), (255, 255, 255))
    mask2 = cv2.inRange(img_hsv, (70, 100, 0), (255, 255, 255))
    mask = mask1 + mask2
    kernel = np.ones((1, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_blur = cv2.medianBlur(mask, 5)
    canny = cv2.Canny(mask_blur, 100, 300, 3)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= limit_area:
            nuclei.append(cnt)
            print(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append([x, y, w, h])
            rects.append([x, y, w, h])

    rects, weights = cv2.groupRectangles(rects, 1, 1.5)
    draw(rects, (0, 0, 255))
    cv2.imshow('img', img)

    k = cv2.waitKey(50) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()