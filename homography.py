import cv2

class Homography:
    def __init__(self):
        pass

    def findHomography(self):
        raise NotImplementedError
    
class DefaultHomography(Homography):
    def __init__(self):
        super().__init__()

    def findHomography(self, src_pts, dst_pts, ransacReprojThreshold=5.0):
        return cv2.findHomography(src_pts, dst_pts, 0, ransacReprojThreshold)

class LMEDSHomography(Homography):
    def __init__(self):
        super().__init__()

    def findHomography(self, src_pts, dst_pts, ransacReprojThreshold=5.0):
        return cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, ransacReprojThreshold)
    
class RANSACHomography(Homography):
    def __init__(self):
        super().__init__()

    def findHomography(self, src_pts, dst_pts, ransacReprojThreshold=5.0):
        return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
    
class USACMAGSACHomography(Homography):
    def __init__(self):
        super().__init__()

    def findHomography(self, src_pts, dst_pts, ransacReprojThreshold=5.0):
        return cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, ransacReprojThreshold)

class RHOHomography(Homography):
    def __init__(self):
        super().__init__()

    def findHomography(self, src_pts, dst_pts, ransacReprojThreshold=5.0):
        return cv2.findHomography(src_pts, dst_pts, cv2.RHO, ransacReprojThreshold)