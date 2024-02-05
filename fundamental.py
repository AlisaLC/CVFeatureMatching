import cv2

class Fundamental:
    def __init__(self):
        pass

    def findFundamental():
        raise NotImplementedError
    
class DefaultFundamental(Fundamental):
    def __init__(self):
        super().__init__()

    def findFundamental(self, src_pts, dst_pts):
        return cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_8POINT)

class RANSACFundamental(Fundamental):
    def __init__(self):
        super().__init__()

    def findFundamental(self, src_pts, dst_pts):
        return cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

class USACMAGSACFundamental(Fundamental):
    def __init__(self):
        super().__init__()

    def findFundamental(self, src_pts, dst_pts):
        return cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC)
    
class LMEDSFundamental(Fundamental):
    def __init__(self):
        super().__init__()

    def findFundamental(self, src_pts, dst_pts):
        return cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)
    
class FM_7POINTFundamental(Fundamental):
    def __init__(self):
        super().__init__()

    def findFundamental(self, src_pts, dst_pts):
        return cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_7POINT)

class USACDEFAULTFundamental(Fundamental):
    def __init__(self):
        super().__init__()

    def findFundamental(self, src_pts, dst_pts):
        return cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_DEFAULT)

class USACPARALLELFundamental(Fundamental):
    def __init__(self):
        super().__init__()

    def findFundamental(self, src_pts, dst_pts):
        return cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_PARALLEL)

class USACFASTFundamental(Fundamental):
    def __init__(self):
        super().__init__()

    def findFundamental(self, src_pts, dst_pts):
        return cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_FAST)

class USACACCURATEFundamental(Fundamental):
    def __init__(self):
        super().__init__()

    def findFundamental(self, src_pts, dst_pts):
        return cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_ACCURATE)       