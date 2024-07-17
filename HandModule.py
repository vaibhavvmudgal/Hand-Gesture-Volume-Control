import cv2 as cv
import mediapipe as mp
import time

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
        #videoRGB = cv.cvtColor(video_data, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(video_data)

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
    
def main():
    cap = cv.VideoCapture(0)
    detector = handDetector()
    cTime = 0
    pTime = 0

    while True:
        success, video_data = cap.read()
        if not success:
            break

        video_data = detector.findHands(video_data)
        lmList = detector.findPosition(video_data)
        if lmList:
            print(lmList[4])
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(video_data, str(int(fps)), (10, 70), cv.FONT_HERSHEY_DUPLEX, 3, (0, 255, 255), 4)
        cv.imshow("LIVE", video_data)

        if cv.waitKey(10) == ord("a"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()