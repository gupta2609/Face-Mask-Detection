from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

with_mask = np.load('With_Mask.npy')
without_mask = np.load('WithOut_Mask.npy')

with_mask = with_mask.reshape(200,50*50*3)
without_mask = without_mask.reshape(200,50*50*3)

rows = np.r_[with_mask,without_mask]
labels = np.zeros(rows.shape[0])
labels[200:]=1.0
flags = {0:'MASK',1:'WITHOUT MASK'}

x_train,x_test,y_train,y_test=train_test_split(rows,labels,test_size=0.20)
svm = SVC()
svm.fit(x_train,y_train)

y_pred=svm.predict(x_test)

haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recorder = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    flag,fram = recorder.read()
    if flag:
        faces = haar_data.detectMultiScale(fram)
        for x,y,w,h in faces:
            cv2.rectangle(fram,(x,y),(x+w,y+h),(0,255,255),4)
            face = fram[y:y+h,x:x+w,:]
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            pred = svm.predict(face)[0]
            res = flags[int(pred)]
            cv2.putText(fram,res,(x,y),font,1,(255,255,0),2)
            print(res)
        cv2.imshow('output',fram)
        if cv2.waitKey(2)==27:
            break
recorder.release()
cv2.destroyAllWindows()


