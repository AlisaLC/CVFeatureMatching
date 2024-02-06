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

method = st.sidebar.selectbox("Choose Method", ["Classic", "Deep"])

if method == "Classic":
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
else:
    st.sidebar.title("Deep Matcher")
    deep_matcher = st.sidebar.selectbox("Choose a deep matcher", ["LoFTR"])
    deep_matchers = {
        "LoFTR": LoFTRMatcher
    }

    st.sidebar.title("Fundamental Matrix")
    fundamental = st.sidebar.selectbox("Choose a fundamental matrix method", ["RANSAC", "Default", "USAC_MAGSAC", "LMEDS", "FM_7POINT", "USAC_DEFAULT", "USAC_PARALLEL", "USAC_FAST", "USAC_ACCURATE"])
    fundamentals = {
        "RANSAC": RANSACFundamental,
        "Default": DefaultFundamental,
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

metrics = {
    "Keypoints in Image 1": 0,
    "Keypoints in Image 2": 0,
    "Matches Found": 0
}

uploaded_files = st.file_uploader("Choose an image...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if len(uploaded_files) == 2:
    img1, img2 = uploaded_files
    img1 = load_image_from_bytes(img1.getvalue())
    img1_rgb = to_rgb(img1)
    gray1 = to_gray(img1)
    img2 = load_image_from_bytes(img2.getvalue())
    img2_rgb = to_rgb(img2)
    gray2 = to_gray(img2)

    col1, col2 = st.columns(2)
    col1.image(img1_rgb, use_column_width=True, caption="Image 1")
    col2.image(img2_rgb, use_column_width=True, caption="Image 2")

    with timer("Total Processing Time"):
        if method == "Deep":
            fundamental = fundamentals[fundamental]()
            deep_matcher = deep_matchers[deep_matcher]()

            scale = 1000 / max(img1.shape[:2])
            img1 = cv2.resize(img1, (0, 0), fx=scale, fy=scale)
            img1_rgb = to_rgb(img1)
            gray1 = to_gray(img1)
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            img2_rgb = to_rgb(img2)
            gray2 = to_gray(img2)

            with timer("Detect and Match Time"):
                kp1, kp2 = deep_matcher(gray1, gray2)
                matches = [(i, i) for i in range(len(kp1))]

            with timer("Plotting keypoints"):
                kp1_cv = [cv2.KeyPoint(x, y, 3) for x, y in kp1.reshape(-1, 2)]
                kp2_cv = [cv2.KeyPoint(x, y, 3) for x, y in kp2.reshape(-1, 2)]
                img1_kp = draw_keypoints(img1_rgb, kp1_cv)
                img2_kp = draw_keypoints(img2_rgb, kp2_cv)
                col1.image(img1_kp, use_column_width=True, caption="Image 1")
                col2.image(img2_kp, use_column_width=True, caption="Image 2")
            
            F, inliers = fundamental.findFundamental(kp1, kp2)
            inliers = inliers > 0

            with timer("Plotting Time"):
                draw_loftr_matches(img1_rgb, img2_rgb, kp1, kp2, inliers)
                plt.axis('off')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
                buffer.seek(0)
            
            img = load_image_from_bytes(buffer.getvalue())
            img_rgb = to_rgb(img)
            st.image(img_rgb, use_column_width=True)
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
            
            with timer("Plotting keypoints"):
                img1_kp = draw_keypoints(img1_rgb, kp1)
                img2_kp = draw_keypoints(img2_rgb, kp2)
                col1.image(img1_kp, use_column_width=True, caption="Image 1")
                col2.image(img2_kp, use_column_width=True, caption="Image 2")

            with timer("Matcher Processing Time"):
                matches = matcher(des1, des2)
                matches = matcher.filter_matches(matches)

            st.image(draw_matches(img1_rgb, img2_rgb, kp1, kp2, matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

    timing_data = [{"Process Type": key, "Time To Take (Miliseconds)": value} for key, value in timing_results.items()]
    st.table(timing_data)

    metrics["Keypoints in Image 1"] = len(kp1)
    metrics["Keypoints in Image 2"] = len(kp2)
    metrics["Matches Found"] = len(matches)
    
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label=label, value=value)