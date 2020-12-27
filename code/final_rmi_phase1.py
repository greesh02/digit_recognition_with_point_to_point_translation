import cv2, numpy as np
import time

# PHASE 1 ############################################################################################################################################################

cap = cv2.VideoCapture(0)
w = int(cap.get(3))
h = int(cap.get(4))
print(w, h)
black = np.ones((h, w, 3), np.uint8)
draw = np.ones((h, w, 3), np.uint8)
# draw = np.full((h,w,3),255,dtype=np.uint8)
minus_img = np.ones((h, w, 3), np.uint8)

k = 0
kk = 0
t = 0

pts = []
timer = 0


def time_up():
    global k, pts
    k = 0
    pts = []


kernel = np.ones((5, 5), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    frame1 = frame.copy()
    blur_in = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur_in, cv2.COLOR_BGR2HSV)
    black_flip = cv2.flip(black, 1)

    g = np.uint8([[[0, 255, 0]]])
    green_l = np.array([50, 80, 100])
    green_h = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, green_l, green_h)
    blur = cv2.medianBlur(mask, 5)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    adapth = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(adapth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # print(len(contours),"kkkk")

    cv2.drawContours(frame1, contours, -1, (255, 0, 0), 3)

    for contour in contours:
        if 500 < cv2.contourArea(contour) < 1500:
            print("in")
            (x, y), radius = cv2.minEnclosingCircle(contour)
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 3)
            pts.append((int(x), int(y)))

            t = 0
            if k > 0:
                cv2.line(draw, pts[k - 1], pts[k], (255, 255, 255), 3,
                         cv2.LINE_AA)  ###############################################################################################################
                cv2.line(black, pts[k - 1], pts[k], (255, 255, 0), 3)
                timer = round(time.time())
                k += 1
            if k == 0:
                k += 1
            print(k)
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), 2)
            # cv2.circle(black, (int(x), int(y)),8, (255,255, 255), -1)
            break
        else:

            if round(time.time()) == (timer + 2) and t == 0:
                print("time up", timer, time.time())
                t = 1
                k = 0
                pts = []

    flip = cv2.flip(frame, 1)
    fin = cv2.add(flip, black_flip)
    cv2.line(minus_img, (0, 240), (640, 240), (0, 0, 255), 2)

    fin = cv2.addWeighted(fin, 0.7, minus_img, 0.3, 0)
    draw1 = cv2.flip(draw, 1)

    # print(green_h,green_l)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    # cv2.imshow("open",open)
    # cv2.imshow("erode", erode)
    cv2.imshow("blur", blur)
    cv2.imshow("adapt_thresh", adapth)
    cv2.imshow("flip", flip)
    cv2.imshow("bflip", black_flip)
    cv2.imshow("final", fin)
    cv2.imshow("finak", draw1)

    # cv2.imshow("dilate", dilate)
    # cv2.imshow("frame1", frame1)

    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

cap.release()

print(pts)

cv2.destroyAllWindows()

cv2.imwrite("digit.png", draw1)
