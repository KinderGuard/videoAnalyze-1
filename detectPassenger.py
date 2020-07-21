import cv2
import imutils

cap = cv2.VideoCapture('./test_video1.mp4')
count=0
while(1):
    # Initializing the HOG person
    # detector
    ret,frame=cap.read()
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cv2.imwrite("./images2/frame%d.jpg" % count, frame)

    # Reading the Image
    image = cv2.imread(r"./images2/frame%d.jpg" % count)
    count = count + 1

    # Resizing the Image
    image = imutils.resize(image, width=min(400, image.shape[1]))

    # Detecting all the regions in the
    # Image that has a pedestrians inside it
    (regions, _) = hog.detectMultiScale(image,
                                      winStride=(4, 4),
                                        padding=(4, 4),
                                        scale=1.05)

    # Drawing the regions in the Image
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y),
                      (x + w, y + h),
                      (0, 0, 255), 2)

    # Showing the output Image
    cv2.imshow("Image", image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()