
import numpy as np
import cv2
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('vid3.mp4')
cap.set(3,1280) # set Width
cap.set(4,720) # set Height
cap.set(cv2.CAP_PROP_FPS,60)
while True:
    ret,img=cap.read()
    fps=cap.get(5)
    print(fps)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        r=int(h/4)
        
        
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img,(x+20,y),(x+w-20,y+r),(0,0,255),2)
        cv2.putText(img,'Dharma',(x-10,y-5),0,1,(0,255,0))
        
        forehead_roi = img[y:y+r, x:x+w]
          
    cv2.imshow('video',img)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()