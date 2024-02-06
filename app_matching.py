import streamlit as st
from detector import *
from matcher import *
from fundamental import *
from deep import *
from utils import load_image_from_bytes, to_gray, to_rgb, draw_keypoints, draw_matches, warp_images, draw_loftr_matches
from contextlib import contextmanager
import cv2
import matplotlib.pyplot as plt
import io
import time

st.title("Image Matching")

st.sidebar.title("Detector")
detector = st.sidebar.selectbox("Choose a detector", ["SIFT", "SURF", "FAST", "BRIEF", "ORB", "MSER", "AKAZE", "BRISK"])
detectors = {
    "SIFT": SIFTDetector,
    "SURF": SURFDetector,
    "FAST": FastDetector,
    "BRIEF": BRIEFDetector,
    "ORB": ORBDetector,
    "MSER": MSERDetector,
    "AKAZE": AKAZEDetector,
    "BRISK": BRISKDetector
}

st.sidebar.title("Matcher")
matcher = st.sidebar.selectbox("Choose a matcher", ["BFMatcher", "FLANNMatcher"])
matchers = {
    "BFMatcher": BFMatcher,
    "FLANNMatcher": FLANNMatcher
}

st.sidebar.title("Deep Matcher")
deep_matcher = st.sidebar.selectbox("Choose a deep matcher", ["None", "LoFTR"])
deep_matchers = {
    "None": None,
    "LoFTR": LoFTRMatcher
}

st.sidebar.title("Fundamental Matrix")
fundamental = st.sidebar.selectbox("Choose a fundamental matrix method", ["None", "Default", "RANSAC", "USAC_MAGSAC", "LMEDS", "FM_7POINT", "USAC_DEFAULT", "USAC_PARALLEL", "USAC_FAST", "USAC_ACCURATE"])
fundamentals = {
    "None": None,
    "Default": DefaultFundamental,
    "RANSAC": RANSACFundamental,
    "USAC_MAGSAC": USACMAGSACFundamental,
    "LMEDS": LMEDSFundamental,
    "FM_7POINT": FM_7POINTFundamental,
    "USAC_DEFAULT": USACDEFAULTFundamental,
    "USAC_PARALLEL": USACPARALLELFundamental,
    "USAC_FAST": USACFASTFundamental,
    "USAC_ACCURATE": USACACCURATEFundamental
}

timing_results = {}

@contextmanager
def timer(label):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed = (end - start) * 1000
        timing_results[label] = f"{elapsed:.2f} ms"


uploaded_files = st.file_uploader("Choose an image...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if len(uploaded_files) == 2:
    img1, img2 = uploaded_files
    img1 = load_image_from_bytes(img1.getvalue())
    gray1 = to_gray(img1)
    img2 = load_image_from_bytes(img2.getvalue())
    gray2 = to_gray(img2)

    # plot images in 2 columns side by side
    col1, col2 = st.columns(2)
    col1.header("Image 1")
    col1.image(img1, use_column_width=True)
    col2.header("Image 2")
    col2.image(img2, use_column_width=True)

    with timer("Total Processing Time"):

        if deep_matcher != "None":
            if fundamental == "None":
                fundamental = "RANSAC"
            fundamental = fundamentals[fundamental]()
            deep_matcher = deep_matchers[deep_matcher]()
            # downsize img1 so width and height are less than 1000
            scale = 1000 / max(img1.shape[:2])
            img1 = cv2.resize(img1, (0, 0), fx=scale, fy=scale)
            gray1 = to_gray(img1)
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            gray2 = to_gray(img2)
            kp1, kp2 = deep_matcher(gray1, gray2)
            F, inliers = fundamental.findFundamental(kp1, kp2)
            inliers = inliers > 0
            draw_loftr_matches(img1, img2, kp1, kp2, inliers)
            plt.axis('off')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            buffer.seek(0)
            img = load_image_from_bytes(buffer.getvalue())
            st.image(img, use_column_width=True)
        else:
            detector = detectors[detector]()
            matcher = matchers[matcher]()

            with timer("Detector Processing Time"):

                kp1 = detector.detect(gray1)
                kp1 = detector.filter_points(kp1)
                kp1, des1 = detector.compute(gray1, kp1)
                kp2 = detector.detect(gray2)
                kp2 = detector.filter_points(kp2)
                kp2, des2 = detector.compute(gray2, kp2)

            with timer("Matcher Processing Time"):

                matches = matcher(des1, des2)
                matches = matcher.filter_matches(matches)

            st.image(draw_matches(img1, img2, kp1, kp2, matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
    
    timing_data = [{"Process Type": key, "Time To Take (Miliseconds)": value} for key, value in timing_results.items()]
    st.table(timing_data)