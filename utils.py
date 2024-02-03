import cv2

def load_image(image_path):
    return cv2.imread(image_path)

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def draw_keypoints(image, keypoints):
    return cv2.drawKeypoints(image, keypoints, None)

def draw_matches(image1, image2, keypoints1, keypoints2, matches):
    return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)