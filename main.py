import cv2
import numpy as np

haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recorder = cv2.VideoCapture(0)
faces_data = []
while True:
  flag,fram = recorder.read()
  if flag:
    faces = haar_data.detectMultiScale(fram)
    for x,y,w,h in faces:
      cv2.rectangle(fram,(x,y),(x+w,y+h),(0,255,255),4)
      face = fram[y:y+h,x:x+w,:]
      face = cv2.resize(face,(50,50))
      print(len(faces_data))
      if len(faces_data)<200:
        faces_data.append(face)
    cv2.imshow('output',fram)
    if cv2.waitKey(2)==27 or len(faces_data)>=200:
      break
recorder.release()
cv2.destroyAllWindows()

#np.save('WithOut_Mask.npy',faces_data)
np.save('With_Mask',faces_data)