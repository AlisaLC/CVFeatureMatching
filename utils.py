import cv2

def load_image(image_path):
    return cv2.imread(image_path)

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def draw_keypoints(image, keypoints, **kwargs):
    return cv2.drawKeypoints(image, keypoints, None, **kwargs)

def draw_matches(image1, image2, keypoints1, keypoints2, matches, **kwargs):
    return cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None, **kwargs)