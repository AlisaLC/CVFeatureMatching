import streamlit as st
from detector import *
from matcher import *
from homography import *
from utils import load_image_from_bytes, to_gray, to_rgb, draw_keypoints, draw_matches, warp_images

st.title("Image Stitching")

st.sidebar.title("Detector")
detector = st.sidebar.selectbox("Choose a detector", ["SIFT", "SURF", "FAST", "BRIEF", "ORB"])
detectors = {
    "SIFT": SIFTDetector,
    "SURF": SURFDetector,
    "FAST": FastDetector,
    "BRIEF": BRIEFDetector,
    "ORB": ORBDetector
}

st.sidebar.title("Matcher")
matcher = st.sidebar.selectbox("Choose a matcher", ["BFMatcher", "FLANNMatcher"])
matchers = {
    "BFMatcher": BFMatcher,
    "FLANNMatcher": FLANNMatcher
}

st.sidebar.title("Homography")
homography = st.sidebar.selectbox("Choose a homography", ["RANSACHomography", "RHOHomography", "LMEDSHomography", "USACMAGSACHomography", "DefaultHomography"])
homographies = {
    "RANSACHomography": RANSACHomography,
    "RHOHomography": RHOHomography,
    "LMEDSHomography": LMEDSHomography,
    "USACMAGSACHomography": USACMAGSACHomography,
    "DefaultHomography": DefaultHomography
}

uploaded_files = st.file_uploader("Choose an image...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if len(uploaded_files) > 1:
    
    detector = detectors[detector]()
    matcher = matchers[matcher]()
    homography = homographies[homography]()

    imgs = [to_rgb(load_image_from_bytes(file.getvalue())) for file in uploaded_files][::-1]
    img = imgs[0]

    for i in range(1, len(imgs)):
        img_i = imgs[i]
        gray = to_gray(img)
        gray_i = to_gray(img_i)
        keypoints = detector.detect(gray)
        keypoints = detector.filter_points(keypoints)
        keypoints, descriptors = detector.compute(gray, keypoints)
        keypoints_i = detector.detect(gray_i)
        keypoints_i = detector.filter_points(keypoints_i)
        keypoints_i, descriptors_i = detector.compute(gray_i, keypoints_i)
        matches = matcher.knnMatch(descriptors, descriptors_i)
        matches = matcher.filter_matches(matches)
        src_pts, dst_pts = matcher.get_points(keypoints, keypoints_i, matches)
        H, _ = homography.findHomography(dst_pts, src_pts)
        img = warp_images(img, img_i, H)
    st.image(img, caption=f"Stitched Image", use_column_width=True)
else:
    st.write("Please upload at least two images to stitch.")