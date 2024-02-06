import streamlit as st
from contextlib import contextmanager
from utils import load_image_from_bytes, to_gray, draw_keypoints
from detector import *
from deep import *
import time

st.title("Image Keypoint Detection")

detector_options = ["SIFT", "FAST", "BRIEF", "ORB", "Harris", "Shi-Tomasi", "MSER", "AKAZE", "BRISK"]

st.sidebar.title("Options")
selected_detector = st.sidebar.selectbox("Choose a detector", detector_options)

detectors = {
    "SIFT": SIFTDetector,
    "FAST": FastDetector,
    "BRIEF": BRIEFDetector,
    "ORB": ORBDetector,
    "Harris": HarrisDetector,
    "Shi-Tomasi": ShiTomasiDetector,
    "MSER": MSERDetector,
    "AKAZE": AKAZEDetector,
    "BRISK": BRISKDetector,
}

timing_results = {}

@contextmanager
def timer(label):
    start = time.time()
    yield
    end = time.time()
    timing_results[label] = f"{(end - start) * 1000:.2f} ms"
    # st.sidebar.table([[label, f"{(end - start) * 1000:.2f} ms"]])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:

    image = load_image_from_bytes(uploaded_file.getvalue())

    # create 2 columns
    col1, col2 = st.columns(2)

    col1.image(image, caption="Uploaded Image", use_column_width=True)
    
    gray_image = to_gray(image)

    with timer("Detection Process"):
        detector = detectors[selected_detector]()
        keypoints = detector.detect(gray_image)

    keypoints_image = draw_keypoints(image, keypoints)
    col2.image(keypoints_image, caption="Detected Keypoints", use_column_width=True)

    timing_data = [{"Process Type": key, "Time To Take (Miliseconds)": value} for key, value in timing_results.items()]
    st.sidebar.table(timing_data)

    st.sidebar.metric(label="Number of Keypoints", value=len(keypoints))