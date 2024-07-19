import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import os

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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = handDetector()
        self.minVol = 0
        self.maxVol = 100
        self.volBar = 300

    def set_volume(self, volume_level):
        """Set the system volume on macOS."""
        osascript_command = f"osascript -e 'set volume output volume {volume_level}'"
        os.system(osascript_command)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = self.detector.findHands(img)
        lmList = self.detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            length = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [25, 200], [self.minVol, self.maxVol])
            self.volBar = np.interp(length, [25, 200], [400, 150])
            self.set_volume(vol)

            if length <= 50:
                cv.circle(img, (cx, cy), 15, (0, 255, 0), cv.FILLED)

        cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv.rectangle(img, (50, int(self.volBar)), (85, 400), (0, 255, 0), cv.FILLED)

        return img

def main():
    st.title("Hand Gesture Volume Control")

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
