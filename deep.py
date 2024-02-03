from kornia.feature import LoFTR
import torch

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