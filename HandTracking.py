import cv2
import mediapipe as mp
#model kao hands
class handDetector():
    def __init__(self,mode=False,maxHands=1,detectionCon=0,trackCon=0.5):
        self.mode = mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands #model Hands
        self.mpDraw = mp.solutions.drawing_utils #model drawing utils
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionCon,self.trackCon)


    def findHand(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # objekat hands samo rgb prima

        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks: # ova kolekcija sadrzi ruku i njene landmarke
            if draw:
             handLms =self.results.multi_hand_landmarks[0]
             self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img #vraca sliku orginal a ne imgRGB jer je ona cropovana


    def findPosition(self,img, handNo=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[0] #jer je dozvoljena samo jedna ruka u klasi pa je index 0
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm) u koordinatama
                visina,sirina,chanels = img.shape
                cx, cy =int (lm.x * sirina),int (lm.y * visina)  #u pikselima
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy) , 7, (200,0,0),cv2.FILLED)

        return lmList
