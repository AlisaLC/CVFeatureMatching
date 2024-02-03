import cv2
import numpy as np
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import torch
from kornia_moons.viz import draw_LAF_matches

def load_image(image_path):
    return cv2.imread(image_path)

def load_image_from_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def draw_keypoints(image, keypoints, **kwargs):
    return cv2.drawKeypoints(image, keypoints, None, **kwargs)

def draw_matches(image1, image2, keypoints1, keypoints2, matches, **kwargs):
    return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, **kwargs)

def warp_images(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1
    return output_img

def draw_loftr_matches(image1, image2, keypoints1, keypoints2, inliers):
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(
            torch.from_numpy(keypoints1).view(1, -1, 2),
            torch.ones(keypoints1.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypoints1.shape[0]).view(1, -1, 1),
        ),
        KF.laf_from_center_scale_ori(
            torch.from_numpy(keypoints2).view(1, -1, 2),
            torch.ones(keypoints2.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypoints2.shape[0]).view(1, -1, 1),
        ),
        torch.arange(keypoints1.shape[0]).view(-1, 1).repeat(1, 2),
        image1,
        image2,
        inliers,
        draw_dict={
            "inlier_color": (0.2, 1, 0.2),
            "tentative_color": (1.0, 0.5, 1),
            "feature_color": (0.2, 0.5, 1),
            "vertical": False,
        },
    )