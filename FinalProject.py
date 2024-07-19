import cv2 as cv
import numpy as np
import HandModule as hm
import math
import streamlit as st
import time

def main():
    st.title("Hand Gesture Control")

    start_button = st.button("Start")

    if start_button:
        run_camera()

def run_camera():
    wCam, hCam = 640, 480
    video_cap = cv.VideoCapture(0)
    video_cap.set(3, wCam)
    video_cap.set(4, hCam)

    detect = hm.handDetector()

    volBar = 300

    cTime = 0
    pTime = 0

    stframe = st.empty()

    while True:
        ret, video_data = video_cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        try:
            video_data = detect.findHands(video_data)
            lmList = detect.findPosition(video_data, draw=False)
            if len(lmList) != 0:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv.circle(video_data, (x1, y1), 15, (255, 0, 255), cv.FILLED)
                cv.circle(video_data, (x2, y2), 15, (255, 0, 255), cv.FILLED)
                cv.line(video_data, (x1, y1), (x2, y2), (255, 0, 255), 3)

                length = math.hypot(x2 - x1, y2 - y1)
                volBar = np.interp(length, [25, 200], [400, 150])

                if length <= 50:
                    cv.circle(video_data, (cx, cy), 15, (0, 255, 0), cv.FILLED)

            cv.rectangle(video_data, (50, 150), (85, 400), (0, 255, 0), 3)
            cv.rectangle(video_data, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv.putText(video_data, f'FPS: {int(fps)}', (50, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 4)

            # Display the frame in Streamlit
            stframe.image(video_data, channels="BGR")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

        if st.button("Stop"):
            break

    video_cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
