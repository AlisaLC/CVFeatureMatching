import streamlit as st
from contextlib import contextmanager
from utils import load_image_from_bytes, to_rgb, draw_superpoint_matches
from deep import *
import time

st.title("Image Keypoint Detection")

st.sidebar.title("Options")
feature_extractor = st.sidebar.selectbox("Choose a feature extractor", [
    "disk",
    "superpoint_aachen",
    "superpoint_max",
    "superpoint_inloc",
    "r2d2",
    "d2net-ss",
    "sift",
    "sosnet",
    "dir",
    "netvlad",
    "openibl",
    "eigenplaces",
])
matcher = st.sidebar.selectbox("Choose a matcher", [
    "disk+lightglue",
    "superpoint+lightglue",
    "superglue",
    "superglue-fast",
    "NN-superpoint",
    "NN-ratio",
    "NN-mutual",
    "adalam",
])

st.sidebar.title("Metrics")

timing_results = {}

@contextmanager
def timer(label):
    start = time.time()
    yield
    end = time.time()
    timing_results[label] = f"{(end - start) * 1000:.2f} ms"
    # st.sidebar.table([[label, f"{(end - start) * 1000:.2f} ms"]])

uploaded_files = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if len(uploaded_files) > 1:
    imgs = [load_image_from_bytes(file.getvalue()) for file in uploaded_files]
    dir_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    streamlit_path = Path('streamlit')
    try:
        streamlit_path.mkdir()
    except:
        pass
    images_path = streamlit_path / 'images'
    try:
        images_path.mkdir()
    except:
        pass
    dir_path = images_path / dir_name
    dir_path.mkdir()
    mapping = dir_path / 'mapping'
    mapping.mkdir()
    output_path = streamlit_path / 'outputs'
    try:
        output_path.mkdir()
    except:
        pass
    for i, img in enumerate(imgs):
        cv2.imwrite(str(mapping / f'{i}.jpg'), img)
    model = SuperMatcher(feature_extractor, matcher)
    model.init(dir_path, output_path)
    with timer("Feature extraction time"):
        model.feature_extraction()
    with timer("Pair searching time"):
        model.pair_searching()
    with timer("Pair matching time"):
        model.pairs_matching()
    with timer("Reconstruction time"):
        model = model.reconstruction_from_matches()
    with timer("Plotting reconstruction time"):
        fig = draw_superpoint_matches(model)
        st.plotly_chart(fig)

    timing_data = [{"Process Type": key, "Time To Take (Miliseconds)": value} for key, value in timing_results.items()]
    st.table(timing_data)
