import cv2
import numpy as np

img = cv2.imread('smarties.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(imgray, 50, 200)
Hough = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, 10, param1=200, param2=100, minRadius=0, maxRadius=0)

detected_circle = np.uint16(np.around(Hough))
for x, y, r in detected_circle[0,:]:
    cv2.circle(img, (x, y), r, (0, 255, 0), 3)
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()