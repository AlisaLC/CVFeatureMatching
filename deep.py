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
    def __init__(self):
        self.feature_conf = extract_features.confs['superpoint_aachen']
        self.matcher_conf = match_features.confs['superglue']
        self.outputs = Path('outputs/demo/')
        self.sfm_pairs = self.outputs / 'pairs-sfm.txt'
        self.loc_pairs = self.outputs / 'pairs-loc.txt'
        self.sfm_dir = self.outputs / 'sfm'
        self.features = self.outputs / 'features.h5'
        self.matches = self.outputs / 'matches.h5'

    def __call__(self, imgs):
        # save imgs to a random directory
        dir_name = 'img_' + str(''.join(random.choices(string.ascii_uppercase + string.digits, k=6)))
        img_dir = self.outputs / dir_name
        img_dir.mkdir(parents=True, exist_ok=True)
        img_paths = []
        images = Path('datasets/sacre_coeur')
        references = [str(p.relative_to(images)) for p in (images / 'mapping/').iterdir()]
        extract_features.main(self.feature_conf, images, image_list=references, feature_path=self.features)
        pairs_from_exhaustive.main(self.sfm_pairs, image_list=references)
        match_features.main(self.matcher_conf, self.sfm_pairs, features=self.features, matches=self.matches)
        model = reconstruction.main(self.sfm_dir, images, self.sfm_pairs, self.features, self.matches, image_list=references)
        return model