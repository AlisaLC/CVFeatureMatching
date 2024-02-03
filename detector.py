from typing import Any
import cv2
import numpy as np

class KeypointDetector:
    def __init__(self):
        pass

    def __call__(self, *args: Any, **kwds: Any):
        return self.detect_and_compute(*args, **kwds)

    def detect(self, image):
        raise NotImplementedError
    
    def compute(self, image, keypoints):
        raise NotImplementedError
    
    def detect_and_compute(self, image):
        raise NotImplementedError
    
    def filter_points(self, keypoints, min_size=3):
        return [kp for kp in keypoints if kp.size > min_size]

    
class HarrisDetector(KeypointDetector):
    def __init__(self):
        pass

    def __harris_corners(self, grayscaled_image):
        image = grayscaled_image.astype(np.float32)
        dst = cv2.cornerHarris(image, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
        dst = np.uint8(dst)
        return dst
    
    def detect(self, image):
        return self.__harris_corners(image)
    
    def compute(self, image, keypoints):
        return None, None
    
    def detect_and_compute(self, image):
        return self.__harris_corners(image), None

class ShiTomasiDetector(KeypointDetector):
    def __init__(self):
        pass
    
    def detect(self, image):
        return self.__shi_tomasi_corners(image)
    
    def compute(self, image, keypoints):
        return None, None
    
    def detect_and_compute(self, image):
        return self.__shi_tomasi_corners(image), None
    
    def __shi_tomasi_corners(self, grayscaled_image, max_corners=23):
        corners = cv2.goodFeaturesToTrack(grayscaled_image, max_corners, 0.01, 10)
        corners = np.int0(corners)
        return corners

class SIFTDetector(KeypointDetector):
    def __init__(self):
        self.detector = cv2.SIFT_create()
    
    def detect(self, image):
        return self.detector.detect(image)
    
    def compute(self, image, keypoints):
        return self.detector.compute(image, keypoints)
    
    def detect_and_compute(self, image):
        return self.detector.detectAndCompute(image, None)

class SURFDetector(KeypointDetector):
    def __init__(self):
        self.detector = cv2.xfeatures2d.SURF_create()
    
    def detect(self, image):
        return self.detector.detect(image)
    
    def compute(self, image, keypoints):
        return self.detector.compute(image, keypoints)
    
    def detect_and_compute(self, image):
        return self.detector.detectAndCompute(image, None)

class FastDetector(KeypointDetector):
    def __init__(self):
        self.detector = cv2.FastFeatureDetector_create()
    
    def detect(self, image):
        return self.detector.detect(image)
    
    def compute(self, image, keypoints):
        return None, None
    
    def detect_and_compute(self, image):
        return self.detector.detect(image), None

class BRIEFDetector(KeypointDetector):
    def __init__(self):
        self.star = cv2.xfeatures2d.StarDetector_create()
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    
    def detect(self, image):
        return self.star.detect(image)
    
    def compute(self, image, keypoints):
        return self.brief.compute(image, keypoints)
    
    def detect_and_compute(self, image):
        keypoints = self.star.detect(image)
        return keypoints, self.brief.compute(image, keypoints)

class ORBDetector(KeypointDetector):
    def __init__(self):
        self.detector = cv2.ORB_create()
    
    def detect(self, image):
        return self.detector.detect(image)
    
    def compute(self, image, keypoints):
        return self.detector.compute(image, keypoints)
    
    def detect_and_compute(self, image):
        return self.detector.detectAndCompute(image, None)