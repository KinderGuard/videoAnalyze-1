import numpy as np
from scipy import signal
import cv2


def draw_flow(im, flow, step=8):
    h, w = im.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    # create line endpoints
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    # create image and draw
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) < 2:
            continue
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(vis, (x2, y2), 1, (0, 0, 255), 2)
    return vis


def get_flow(U, V, im):
    h, w = im.shape[:2]
    flow = np.zeros((h, w, 2))
    for i in range(h):
        for j in range(w):
            flow[i][j][0] = U[i][j]
            flow[i][j][1] = V[i][j]
    return flow


def optical_flow(I1g, I2g, window_size=7, tau=1e-2):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    w = window_size / 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t,
                                                                                          boundary='symm', mode=mode)

    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0] - w):
        for j in range(w, I1g.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
            b = np.reshape(It, (It.shape[0], 1))
            A = np.vstack((Ix, Iy)).T
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                u[i, j] = nu[0]
                v[i, j] = nu[1]

    return u, v


cap = cv2.VideoCapture('./test_video1.mp4')

ret, im = cap.read()
prev_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

while True:
    # get grayscale image
    ret, im = cap.read()
    if ret:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # compute flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) #sparse로 바꿔야함!!
        prev_gray = gray
        # plot the flow vectors
        cv2.imshow('Optical flow', draw_flow(gray, flow))

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()


# u, v = optical_flow(prev_gray, gray, 21)
# flow = get_flow(u, v, im)
# vis = draw_flow(gray, flow, step=16)
# cv2.imwrite("LK_cubic.png", vis)
