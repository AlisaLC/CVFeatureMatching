import random
import string
import cv2
from kornia.feature import LoFTR
import torch
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LoFTRMatcher:
    def __init__(self, dataset="outdoor"):
        self.loftr = LoFTR(pretrained=dataset).to(DEVICE)
        self.loftr.eval()

    def __call__(self, img1, img2):
        img1 = torch.tensor(img1 / 255.).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
        img2 = torch.tensor(img2 / 255.).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
        with torch.no_grad():
            matches = self.loftr({'image0': img1, 'image1': img2})
        keypoints0 = matches['keypoints0'].cpu().numpy()
        keypoints1 = matches['keypoints1'].cpu().numpy()
        return keypoints0, keypoints1

class SuperMatcher:
    def __init__(self, feature_conf='disk', matcher_conf='disk+lightglue'):
        self.feature_conf = extract_features.confs[feature_conf]
        self.matcher_conf = match_features.confs[matcher_conf]

    def init(self, images_path, output_path):
        if isinstance(images_path, str):
            images_path = Path(images_path)
        self.images = images_path
        self.references = [str(p.relative_to(self.images)) for p in (self.images / 'mapping/').iterdir()]
        print(self.references)
        if isinstance(output_path, str):
            output_path = Path(output_path)
        self.outputs = output_path
        self.sfm_pairs = self.outputs / 'pairs-sfm.txt'
        self.loc_pairs = self.outputs / 'pairs-loc.txt'
        self.sfm_dir = self.outputs / 'sfm'
        self.features = self.outputs / 'features.h5'
        self.matches = self.outputs / 'matches.h5'
    
    def feature_extraction(self):
        extract_features.main(self.feature_conf, self.images, image_list=self.references, feature_path=self.features)
    
    def pair_searching(self):
        pairs_from_exhaustive.main(self.sfm_pairs, image_list=self.references)
    
    def pairs_matching(self):
        match_features.main(self.matcher_conf, self.sfm_pairs, features=self.features, matches=self.matches)
    
    def reconstruction_from_matches(self):
        model = reconstruction.main(self.sfm_dir, self.images, self.sfm_pairs, self.features, self.matches, image_list=self.references)
        return model

    def __call__(self, images_path, output_path):
        self.init(images_path, output_path)
        self.feature_extraction()
        self.pair_searching()
        self.pairs_matching()
        return self.reconstruction_from_matches()