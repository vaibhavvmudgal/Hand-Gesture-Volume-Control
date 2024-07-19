import cv2 as cv
import numpy as np
import HandModule as hm
import math
import streamlit as st

def simulate_volume_control(length):
    """Simulate volume control for cloud deployment."""
    min_length = 25
    max_length = 200
    min_volume = 0
    max_volume = 100

    # Normalize length and calculate volume
    normalized_length = np.clip(length, min_length, max_length)
    volume = np.interp(normalized_length, [min_length, max_length], [min_volume, max_volume])
    
    return volume

def process_frame(frame, detector):
    """Process a single frame for hand gestures and simulate volume control."""
    if frame is None:
        return frame, 50  # Return default volume if frame is None

    try:
        # Convert to RGB for hand detection
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process the frame with hand detector
        processed_frame = detector.findHands(rgb_frame)
        lmList = detector.findPosition(rgb_frame, draw=False)

        volume = 50  # Default volume
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
            volume = simulate_volume_control(length)
            volBar = np.interp(length, [25, 200], [400, 150])

            # Draw volume bar
            cv.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
            cv.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)

        # Convert to BGR for Streamlit
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        return frame, volume
    except Exception as e:
        st.error(f"Error in process_frame: {e}")
        return frame, 50

def main():
    st.title("Hand Gesture Volume Control")

    detector = hm.handDetector()

    # Use Streamlit's webcam component
    webcam = st.camera_input("Capture webcam feed")

    if webcam:
        stframe = st.empty()
        stop_button = st.button('Stop', key='stop_button')

        volume_display = st.empty()
        while True:
            frame = webcam.read()  # Read a frame from the webcam
            if frame is None:
                st.write("Webcam feed ended or cannot read frame.")
                break

            # Process and display the frame
            processed_frame, volume = process_frame(frame, detector)
            stframe.image(processed_frame, channels="BGR", use_column_width=True)

            # Display simulated volume level
            volume_display.write(f"Simulated Volume Level: {volume:.2f}")

            if stop_button:
                st.write("Processing stopped by user.")
                break
    else:
        st.write("Webcam not available.")

if __name__ == "__main__":
    main()
