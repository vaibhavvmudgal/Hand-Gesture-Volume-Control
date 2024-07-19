import cv2 as cv
import numpy as np
import HandModule as hm
import math
import os
import streamlit as st
import tempfile
import time
import youtube_dl
from moviepy.editor import VideoFileClip
import shutil

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def download_youtube_video(url):
    ydl_opts = {
        'format': 'best',
        'outtmpl': tempfile.mktemp(suffix=".mp4"),  # Temporary file path
        'noplaylist': True,
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_file = ydl.prepare_filename(info_dict)
        return video_file
    except Exception as e:
        st.error(f"Failed to download video: {e}")
        return None

def main():
    st.title("Hand Gesture Volume Control")

    # YouTube URL input
    youtube_url = st.text_input("Enter YouTube URL:")

    start_button = st.button("Start")

    if start_button and youtube_url:
        video_path = download_youtube_video(youtube_url)
        if video_path:
            st.write("Processing video...")
            video_file_path = process_video(video_path)
            if video_file_path:
                st.video(video_file_path)

def process_video(video_path):
    detect = hm.handDetector()

    minVol = 0
    maxVol = 100
    volBar = 300
    vol = 50  # Start with 50% volume

    cTime = 0
    pTime = 0

    cap = cv.VideoCapture(video_path)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.close()

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(temp_file.name, fourcc, 30.0, (640, 480))

    while cap.isOpened():
        ret, video_data = cap.read()
        if not ret:
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

            if length <= 50:
                cv.circle(video_data, (cx, cy), 15, (0, 255, 0), cv.FILLED)

        cv.rectangle(video_data, (50, 150), (85, 400), (0, 255, 0), 3)
        cv.rectangle(video_data, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(video_data, f'FPS: {int(fps)}', (50, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 4)
        out.write(video_data)

    cap.release()
    out.release()
    cv.destroyAllWindows()

    return temp_file.name

if __name__ == "__main__":
    main()
