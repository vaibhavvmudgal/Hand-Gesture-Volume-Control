import cv2 as cv
import numpy as np
import HandModule as hm
import math
import os
import streamlit as st
import tempfile
import time
from pytube import YouTube
from moviepy.editor import VideoFileClip

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import absl.logging
absl.logging.set_verbosity(absl.logging.INFO)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def set_volume(volume_level, video_path):
    """Set the volume of the video."""
    clip = VideoFileClip(video_path)
    new_clip = clip.volumex(volume_level / 100)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    new_clip.write_videofile(temp_file.name, codec='libx264')
    return temp_file.name

def main():
    st.title("Hand Gesture Volume Control")

    # YouTube URL input
    youtube_url = st.text_input("Enter YouTube URL:")

    start_button = st.button("Start")

    if start_button and youtube_url:
        video_path = download_youtube_video(youtube_url)
        if video_path:
            run_video(video_path)

def download_youtube_video(url):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        stream.download(output_path=os.path.dirname(temp_file.name), filename=os.path.basename(temp_file.name))
        return temp_file.name
    except Exception as e:
        st.error(f"Failed to download video: {e}")
        return None

def run_video(video_path):
    detect = hm.handDetector()

    minVol = 0
    maxVol = 100
    volBar = 300
    vol = 50  # Start with 50% volume

    cTime = 0
    pTime = 0

    stframe = st.empty()

    cap = cv.VideoCapture(video_path)

    while cap.isOpened():
        ret, video_data = cap.read()
        if not ret:
            st.write("Failed to read from video.")
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
        
        # Display the frame in Streamlit
        stframe.image(video_data, channels="BGR")

        if cv.waitKey(1) == ord("a"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
