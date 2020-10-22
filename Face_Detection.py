import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_classifier = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_face = face_classifier.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in cascade_face:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0, 255, 0), 3)
        cascade_eye = eyes_classifier.detectMultiScale(gray, 1.1)
        for ex,ey,ew,eh in cascade_eye:
            cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (255,0,0), 3)
    cv2.imshow('face', frame)
    if cv2.waitKey(1) == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()
