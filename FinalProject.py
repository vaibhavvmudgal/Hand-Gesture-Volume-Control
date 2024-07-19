import cv2 as cv
import numpy as np
import HandModule as hm
import math
import os
import streamlit as st
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def set_volume(volume_level):
    """Set the system volume on macOS."""
    osascript_command = f"osascript -e 'set volume output volume {volume_level}'"
    os.system(osascript_command)

def get_volume():
    """Get the system volume on macOS."""
    osascript_command = "osascript -e 'output volume of (get volume settings)'"
    return int(os.popen(osascript_command).read().strip())

def main():
    st.title("Hand Gesture Volume Control")
    
    # Camera selection
    camera_options = ["Webcam"]
    camera = st.selectbox("Choose your camera:", camera_options)

    start_button = st.button("Start")

    if start_button:
        run_camera()

def run_camera():
    wCam, hCam = 640, 480
    video_cap = cv.VideoCapture(0)
    video_cap.set(3, wCam)
    video_cap.set(4, hCam)

    detect = hm.handDetector()

    minVol = 0
    maxVol = 100
    volBar = 300
    vol = 0

    cTime = 0
    pTime = 0

    stframe = st.empty()

    while True:
        ret, video_data = video_cap.read()
        if not ret:
            st.write("Failed to read from camera.")
            break
        
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
            vol = np.interp(length, [25, 200], [minVol, maxVol])
            volBar = np.interp(length, [25, 200], [400, 150])
            set_volume(vol)

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

        if cv.waitKey(1) == ord("a"):
            break

    video_cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
