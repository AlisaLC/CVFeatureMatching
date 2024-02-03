import cv2

def init_sift_detector():
    return cv2.SIFT_create()

def detect_keypoints(detector, image):
    return detector.detectAndCompute(image, None)