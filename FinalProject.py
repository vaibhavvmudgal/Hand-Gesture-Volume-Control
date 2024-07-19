import streamlit as st
import HandModule as hm
import cv2 as cv
import numpy as np
import math
import tempfile
import time

def main():
    st.title("Hand Gesture Volume Control")

    # YouTube URL input
    youtube_url = st.text_input("Enter YouTube URL:")

    start_button = st.button("Start")

    if start_button and youtube_url:
        st.write("Displaying video...")
        video_embed_url = f"https://www.youtube.com/embed/{extract_video_id(youtube_url)}"
        st.video(video_embed_url)

        # Placeholder for hand gesture detection (not used directly)
        st.write("Hand gesture detection is not directly supported for embedded YouTube videos.")

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    import re
    pattern = r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/(?:watch\?v=|embed\/|v\/|.+\/)?([^"&?\/\s]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        st.error("Invalid YouTube URL.")
        return ""

if __name__ == "__main__":
    main()
