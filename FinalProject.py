import streamlit as st
import cv2 as cv
import numpy as np
import HandModule as hm
import math
import tempfile
import os

def process_frame(frame, detector):
    """Process a single frame for hand gestures."""
    if frame is None:
        return frame

    # Convert to RGB for hand detection
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # Process the frame with hand detector
    processed_frame = detector.findHands(rgb_frame)
    lmList = detector.findPosition(rgb_frame, draw=False)

    if lmList:
        # Get coordinates of hand landmarks
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw landmarks and lines
        cv.circle(frame, (x1, y1), 15, (255, 0, 255), cv.FILLED)
        cv.circle(frame, (x2, y2), 15, (255, 0, 255), cv.FILLED)
        cv.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [25, 200], [0, 100])
        volBar = np.interp(length, [25, 200], [400, 150])
        
        if length <= 50:
            cv.circle(frame, (cx, cy), 15, (0, 255, 0), cv.FILLED)

        # Draw volume bar
        cv.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
        cv.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)

    # Convert to BGR for Streamlit
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    return frame

def main():
    st.title("Hand Gesture Volume Control")

    # Load video
    st.write("Upload a video file:")
    video_file = st.file_uploader("Choose a video file", type=["mp4", "mov"])

    if video_file is not None:
        # Save the uploaded video to a temporary file
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        st.write("Processing video...")

        detector = hm.handDetector()
        cap = cv.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Failed to open video.")
            return

        stframe = st.empty()

        # Unique key for the stop button to avoid DuplicateWidgetID error
        stop_button = st.button('Stop', key='stop_button')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Video ended or cannot read frame.")
                break

            processed_frame = process_frame(frame, detector)
            
            # Convert to PIL Image and display
            stframe.image(processed_frame, channels="BGR", use_column_width=True)

            # Check if the stop button was pressed
            if stop_button:
                st.write("Processing stopped by user.")
                break

        cap.release()
        os.remove(video_path)  # Clean up the uploaded video file

if __name__ == "__main__":
    main()
