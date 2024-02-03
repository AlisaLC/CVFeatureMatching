import cv2

def init_bf_matcher():
    return cv2.BFMatcher()

def match_keypoints(matcher, descriptors1, descriptors2, k=2):
    return matcher.knnMatch(descriptors1, descriptors2, k=k)

def filter_matches(matches, ratio=0.75):
    return [m for m, n in matches if m.distance < ratio * n.distance]