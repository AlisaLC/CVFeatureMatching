import cv2

class Matcher:
    def __init__(self):
        pass

    def knnMatch(self, descriptors1, descriptors2, k=2):
        raise NotImplementedError
    
    def filter_matches(self, matches, ratio=0.75):
        return [m for m, n in matches if m.distance < ratio * n.distance]

class BFMatcher(Matcher):
    def __init__(self):
        self.matcher = cv2.BFMatcher()

    def knnMatch(self, descriptors1, descriptors2, k=2):
        return self.matcher.knnMatch(descriptors1, descriptors2, k=k)

class FLANNMatcher(Matcher):
    def __init__(self):
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def knnMatch(self, descriptors1, descriptors2, k=2):
        return self.matcher.knnMatch(descriptors1, descriptors2, k=k)