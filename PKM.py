import os
import time
import cv2
import HandTracking as HTM

wCam, hCam = 640, 480


vid = cv2.VideoCapture(0)
vid.set(3,wCam)  # CAP_PROP_FRAME_WID
vid.set(4,hCam)
folderPath = "FingerImages"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList: #imPath ce biti 1.png ili 2.png
    img = cv2.imread(f'{folderPath}/{imPath}')  #FingerImages\1.png
    overlayList.append(img)


detector = HTM.handDetector(detectionCon=0.75)

while (True):


    ret, frame = vid.read()
    frame = detector.findHand(frame)
    lmList= detector.findPosition(frame,draw=False)
    trenutni = -1
    if len(lmList) != 0 :
        if (lmList[8][2] < lmList [6][2]) & (lmList[12][2] < lmList[10][2])  & (lmList[16][2] > lmList[14][2]) :
            trenutni=1

        if (lmList[16][2] < lmList[14][2]) :
            trenutni=2

        if (lmList[4][1] < lmList[3][1]) & (lmList[8][2] > lmList [6][2])  :
            trenutni=0

        h, w, c = overlayList[trenutni].shape #c-channels

        if (trenutni == 0) :
           #ako je trenutno kamen
           frame[0:h,0:w]=overlayList[2] #odigraj paper
        if (trenutni == 1) :
            #ako je trenutno makaze
            frame[0:h, 0:w] = overlayList[0]  # odigraj rock
        if (trenutni == 2) :
            #ako je trenutno papir
            frame[0:h, 0:w] = overlayList[1]  # odigraj makaze


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()