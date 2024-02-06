import streamlit as st
from detector import *
from matcher import *
from homography import *
from utils import load_image_from_bytes, to_gray, to_rgb, draw_keypoints, draw_matches, warp_images
from contextlib import contextmanager
import time

st.title("Image Stitching")

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
    "BRISK": BRISKDetector,
}

st.sidebar.title("Matcher")
matcher = st.sidebar.selectbox("Choose a matcher", ["BFMatcher", "FLANNMatcher"])
matchers = {
    "BFMatcher": BFMatcher,
    "FLANNMatcher": FLANNMatcher
}

st.sidebar.title("Homography")
homography = st.sidebar.selectbox("Choose a homography", ["RANSACHomography", "RHOHomography", "LMEDSHomography", "USACMAGSACHomography", "DefaultHomography", "USACPARALLELHomography", "USACFASTHomography", "USACACCURATEHomography"])
homographies = {
    "RANSACHomography": RANSACHomography,
    "RHOHomography": RHOHomography,
    "LMEDSHomography": LMEDSHomography,
    "USACMAGSACHomography": USACMAGSACHomography,
    "DefaultHomography": DefaultHomography,
    "USACPARALLELHomography": USACPARALLELHomography,
    "USACFASTHomography": USACFASTHomography,
    "USACACCURATEHomography": USACACCURATEHomography
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

metrics = {}

def chunked_metrics(metrics, chunk_size):
    items = list(metrics.items())  # Convert to list for slicing
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

uploaded_files = st.file_uploader("Choose an image...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if len(uploaded_files) > 1:
    
    detector = detectors[detector]()
    matcher = matchers[matcher]()
    homography = homographies[homography]()

    imgs = [to_rgb(load_image_from_bytes(file.getvalue())) for file in uploaded_files][::-1]
    img = imgs[0]

    with timer("Total Processing Time"):

        for i in range(1, len(imgs)):
            img_i = imgs[i]
            gray = to_gray(img)
            gray_i = to_gray(img_i)

            with timer(f"Detector Processing Time For Image {i}"):

                keypoints = detector.detect(gray)
                keypoints = detector.filter_points(keypoints)
                keypoints, descriptors = detector.compute(gray, keypoints)
                keypoints_i = detector.detect(gray_i)
                keypoints_i = detector.filter_points(keypoints_i)
                keypoints_i, descriptors_i = detector.compute(gray_i, keypoints_i)
                metrics[f"Keypoints in Image {i}"] = len(keypoints_i)

            with timer(f"Matcher Processing Time For Image {i}"):

                matches = matcher.knnMatch(descriptors, descriptors_i)
                matches = matcher.filter_matches(matches)
                src_pts, dst_pts = matcher.get_points(keypoints, keypoints_i, matches)
                metrics[f"Matches in Image {i}"] = len(matches)
            
            with timer(f"Finding Homography Processing Time For Image {i}"):

                H, _ = homography.findHomography(dst_pts, src_pts)
                img = warp_images(img, img_i, H)

    img_rgb = to_rgb(img)
    st.image(img_rgb, caption=f"Stitched Image", use_column_width=True)

    timing_data = [{"Process Type": key, "Time To Take (Miliseconds)": value} for key, value in timing_results.items()]
    st.table(timing_data)

    for chunk in chunked_metrics(metrics, 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, chunk):
            with col:
                st.metric(label=label, value=value)
else:
    st.write("Please upload at least two images to stitch.")