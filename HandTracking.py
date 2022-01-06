import cv2
import mediapipe as mp


class HandDetector():

    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, 1, self.detection_confidence, self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        # Tip node ids for 5 fingers, thumb to little finger
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        # Color space conversion
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # Detects hand presence and draws node and connections
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # Drawing hand model
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        self.lm_list = []
        # Checking if hand detected
        if self.results.multi_hand_landmarks:
            # Outputs for a specific hand
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark):
                # converting x,y to integer dimension coords
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
                if draw:
                      cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return self.lm_list

    def fingers_up(self):
        # a list of length 5 which holds 0 or 1 determining which finger is held up
        fingers = []
        # Checking if thumb is up
        if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Loop through index to little finger
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
