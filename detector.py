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

class MSERDetector(KeypointDetector):
    def __init__(self):
        self.mser = cv2.MSER_create()

    def __extract_regions(self, image):
        regions, _ = self.mser.detectRegions(image)
        keypoints = []
        for region in regions:
            if len(region) > 0:
                region = np.array(region)
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                keypoints.append(cv2.KeyPoint(x + w / 2, y + h / 2, max(w, h)))
        return keypoints
    
    def detect(self, image):
        return self.__extract_regions(image)
    
    def compute(self, image, keypoints):
        return None, None
    
    def detect_and_compute(self, image):
        keypoints = self.__extract_regions(image)
        return keypoints, None

class AKAZEDetector(KeypointDetector):
    def __init__(self):
        super().__init__()
        self.akaze = cv2.AKAZE_create()

    def detect(self, image):
        keypoints = self.akaze.detect(image, None)
        return keypoints
    
    def compute(self, image, keypoints):
        keypoints, descriptors = self.akaze.compute(image, keypoints)
        return keypoints, descriptors
    
    def detect_and_compute(self, image):
        keypoints, descriptors = self.akaze.detectAndCompute(image, None)
        return keypoints, descriptors

class BRISKDetector(KeypointDetector):
    def __init__(self):
        super().__init__()
        self.brisk = cv2.BRISK_create()

    def detect(self, image):
        keypoints = self.brisk.detect(image, None)
        return keypoints
    
    def compute(self, image, keypoints):
        keypoints, descriptors = self.brisk.compute(image, keypoints)
        return keypoints, descriptors
    
    def detect_and_compute(self, image):
        keypoints, descriptors = self.brisk.detectAndCompute(image, None)
        return keypoints, descriptors