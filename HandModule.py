import cv2 as cv
import mediapipe as mp

class handDetector:
    def __init__(self, mode=False, maxHands=2, detCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detCon = detCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.detCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, video_data, draw=True):
        videoRGB = cv.cvtColor(video_data, cv.COLOR_BGR2RGB)  # Convert to RGB
        self.results = self.hands.process(videoRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(video_data, handLms, self.mpHands.HAND_CONNECTIONS)

        return video_data

    def findPosition(self, video_data, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = video_data.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(video_data, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        return lmList
