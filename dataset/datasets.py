import numpy as np
import glob
import os
import cv2
import json
import sys
import torch
from torch.utils import data
from typing import Optional

def get_img(img, landmarks):
    x_min = np.min(landmarks, axis=0)[0]
    y_min = np.min(landmarks, axis=0)[1]
    
    x_max = np.max(landmarks, axis=0)[0]
    y_max = np.max(landmarks, axis=0)[1]

    w = x_max - x_min
    h = y_max - y_min
    x_min = int(x_min - w / 5 if x_min - w / 5 > 0 else 0)
    x_max = int(x_max + w / 5 if x_max + w / 5 < img.shape[1] else img.shape[1])
    y_min = int(y_min - h / 5 if y_min - h / 5 > 0 else 0)
    y_max = int(y_max + h / 5 if y_max + h / 5 < img.shape[0] else img.shape[0])
    w = x_max - x_min
    h = y_max - y_min

    n = landmarks.shape[0]
    new_landmarks = np.zeros((n, 2))
    new_landmarks[:, 0] = (landmarks[:, 0] - x_min) / w
    new_landmarks[:, 1] = (landmarks[:, 1] - y_min) / h
    return img[y_min:y_max, x_min:x_max], new_landmarks

def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

def preprocess_unityeyes_image(img, json_data, datasets, input_width, input_height):
    ow = 160
    oh = 96
    # Prepare to segment eye image
    ih, iw = img.shape[:2]
    ih_2, iw_2 = ih/2.0, iw/2.0

    heatmap_w = int(ow/2)
    heatmap_h = int(oh/2)
    
    #img = cv2.resize(im, (im.shape[1]*3, im.shape[0]*3))
    #img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if datasets == 'B':
        gaze    = np.array(json_data['gaze'])
        landmarks  = np.array(json_data['landmarks'])
        left_corner = landmarks[0]
        right_corner = landmarks[4]
        eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
        eye_middle =  landmarks[24].astype(int)
    elif datasets == 'E':
        gaze    = np.array(json_data['gaze_vec'])
        
        left_corner = np.array(json_data['lid_lm_2D'])[0]
        right_corner = np.array(json_data['lid_lm_2D'])[33]
        eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
        eye_middle =  np.mean([np.amin(np.array(json_data['iris_lm_2D']), axis=0), np.amax(np.array(json_data['iris_lm_2D']), axis=0)], axis=0)
        landmarks  = np.concatenate((np.array(json_data['lid_lm_2D']), np.array(json_data['iris_lm_2D']), np.array(json_data['pupil_lm_2D']), eye_middle.reshape(1,2)))
    else:
        print('UnityEyes do not write!!!')
        exit()
    crop_img, lad = get_img(img, landmarks)

    crop_img = cv2.resize(crop_img, (input_width,input_height))
    # if 1:
    #     print(crop_img.shape)
    #     for (x, y) in lad:
    #         color = (0, 255, 0)
    #         cv2.circle(crop_img, (int(round(x*crop_img.shape[1])), int(round(y*crop_img.shape[0]))), 1, color, -1, lineType=cv2.LINE_AA)

    #     #crop_img = cv2.resize(crop_img, (160,96))
    #     cv2.imshow('c', crop_img)
    #     cv2.waitKey(0)
    #     exit()
    return crop_img, lad, gaze

class EyesDataset(data.Dataset):
    def __init__(self, datasets, dataroot, transforms=None, input_width=160, input_height=112):
        self.dataroot = dataroot
        self.datasets = datasets
        self.input_width = input_width
        self.input_height = input_height
        self.transforms = transforms
        if datasets == 'U':
            self.img_paths = glob.glob(os.path.join(dataroot, 'UnityEyes/images') + '/*.jpg')
        elif datasets == 'E':
            self.img_paths = glob.glob(os.path.join(dataroot, 'Eye200W/images') + '/*/*.jpg')
        elif datasets == 'B':
            self.img_paths = glob.glob(os.path.join(dataroot, 'BL_Eye/images') + '/*.jpg')
        self.img_paths = sorted(self.img_paths)
        self.json_paths = []

        for img_path in self.img_paths:
            json_files = img_path.replace('images', 'json').replace('.jpg', '.json')
            self.json_paths.append(json_files)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        full_img = cv2.imread(self.img_paths[index])
        with open(self.json_paths[index]) as f:
            json_data = json.load(f)
        eye, landmarks, gaze = preprocess_unityeyes_image(full_img, json_data, self.datasets, self.input_width, self.input_height)
        if self.transforms:
            eye = self.transforms(eye)
        return eye, landmarks, gaze
    def __len__(self):
        return len(self.img_paths)