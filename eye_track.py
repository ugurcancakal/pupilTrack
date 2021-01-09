'''
ADM Tronics Image Processing Assignment

Author : Ugurcan Cakal

Effort
2101091813 2 hours

'''

import cv2
import numpy as np

cap = cv2.VideoCapture("eye_recording.flv")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[269:795, 537:1416]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, 5, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(threshold,0,255)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)


    if len(contours):
        (x, y, w, h) = cv2.boundingRect(contours[0])
        cv2.drawContours(roi, [contours[0]], -1, (255, 0, 0), 3) 
        # cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        # cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        cv2.circle(roi, (x+int(w/2), y+int(h/2)), 1, (0, 255, 0), 2)
        # cv2.circle(image, center_coordinates, radius, color, thickness)


    cv2.imshow("ROI", gray_roi)
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Canny", edges)
    cv2.imshow("Highlighted", roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
