import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        # Tip node ids for 5 fingers, thumb to little finger
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        # Color space conversion
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # Detects hand presence and draws node and connections
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    #drawing
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition (self, img, handNumber=0, draw=True):
        self.lmList = []
        # Checking if hand detected
        if self.results.multi_hand_landmarks:
            # Outputs for a specific hand
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(myHand.landmark):
                # converting x,y to integer dimension coords
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                      cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        # a list of length 5 which holds 0 or 1 determining which finger is held up
        fingers = []
        # Checking if thumb is up
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Loop through index to little finger
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers