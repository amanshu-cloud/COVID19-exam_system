import cv2
import numpy as np

white = 255*np.ones((512,512,3),np.uint8)
white2 = 255*np.ones((512,512,3),np.uint8)
black = np.zeros((512,512,3),np.uint8)
questions = ["name ?","age?","phone?"]
kernel = np.ones((3,3),np.uint8)

cv2.namedWindow('white')
cv2.namedWindow('black')
sp = 0

for q in questions:
    cap = cv2.VideoCapture(0)
    cv2.putText(white,q,(10,50+sp),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    sp+=50
    cv2.imshow('white',white)
    while(1):
        ret,frame = cap.read()
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        lower = np.array([0,0,0])
        upper = np.array([180,255,40])
       
        mask = cv2.inRange(hsv,lower,upper)
       
        #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 2)
       
        res = cv2.bitwise_and(frame,frame,mask=mask)
        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
       
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
       

    cv2.imwrite('answer.jpg',white)
    cv2.destroyAllWindows()
   
cap.release()