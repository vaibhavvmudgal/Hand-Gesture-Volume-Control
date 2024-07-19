import streamlit as st
import streamlit.components.v1 as components
import cv2 as cv
import numpy as np
import HandModule as hm
import math
import tempfile
import os
from PIL import Image

def webcam_component():
    """Render the HTML component for webcam access."""
    components.html(open('webcam_component.html', 'r').read(), height=540)

def process_frame(frame, detector):
    """Process a single frame for hand gestures."""
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [25, 200], [0, 100])
        volBar = np.interp(length, [25, 200], [400, 150])
        if length <= 50:
            cv.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 15, (0, 255, 0), cv.FILLED)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    return frame

def main():
    st.title("Live Webcam Hand Gesture Control")

    st.write("This application uses your webcam to control the volume based on hand gestures.")
    webcam_component()
    
    detector = hm.handDetector()
    
    # Create a temporary file to save video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.close()

    # Open video capture
    cap = cv.VideoCapture(0)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(temp_file.name, fourcc, 20.0, (640, 480))

    stframe = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame, detector)
        out.write(processed_frame)

        # Convert to PIL Image and display
        stframe.image(processed_frame, channels="BGR", use_column_width=True)

        if st.button('Stop'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
    
    st.write("Video recording saved to", temp_file.name)

if __name__ == "__main__":
    main()
