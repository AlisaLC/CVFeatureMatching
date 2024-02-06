import streamlit as st
from contextlib import contextmanager
from utils import load_image_from_bytes, to_gray, draw_keypoints, draw_loftr_keypoints
from detector import *
from deep import *
import time

st.title("Image Keypoint Detection")

detector_options = ["SIFT", "FAST", "BRIEF", "ORB", "MSER", "AKAZE", "BRISK"]
deep_matcher_options = ["None", "LoFTR"]

st.sidebar.title("Options")
selected_detector = st.sidebar.selectbox("Choose a detector", detector_options)
selected_deep_matcher = st.sidebar.selectbox("Choose a deep matcher", deep_matcher_options)

detectors = {
    "SIFT": SIFTDetector(),
    "FAST": FastDetector(),
    "BRIEF": BRIEFDetector(),
    "ORB": ORBDetector(),
    "MSER": MSERDetector(),
    "AKAZE": AKAZEDetector(),
    "BRISK": BRISKDetector(),
    "LoFTR": LoFTRMatcher()
}

timing_results = {}

@contextmanager
def timer(label):
    start = time.time()
    yield
    end = time.time()
    timing_results[label] = f"{(end - start) * 1000:.2f} ms"

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:

    image = load_image_from_bytes(uploaded_file.getvalue())
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    gray_image = to_gray(image)

    with timer("Detection Process"):
        if selected_deep_matcher == "None":
            detector = detectors[selected_detector]
            keypoints = detector.detect(gray_image)
            keypoints, _ = detector.compute(gray_image, keypoints)
            keypoints_image = draw_keypoints(image, keypoints)
        else:
            detector = detectors[selected_deep_matcher]
            scale = 1000 / max(image.shape[:2])
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            gray1 = to_gray(image)
            keypoints = detector.detect_keypoints(gray1)
            keypoints_image = draw_loftr_keypoints(image, keypoints)

    
    st.image(keypoints_image, caption="Detected Keypoints", use_column_width=True)

    timing_data = [{"Process Type": key, "Time To Take (Miliseconds)": value} for key, value in timing_results.items()]
    st.sidebar.table(timing_data)

    st.sidebar.metric(label="Number of Keypoints", value=len(keypoints))