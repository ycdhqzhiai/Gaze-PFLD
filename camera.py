import argparse

import numpy as np
import cv2

import torch
import torchvision

from models.pfld import Gaze_PFLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(args):

    checkpoint = torch.load(args.model_path, map_location=device)
    print(checkpoint.keys())
    gaze_pfld = Gaze_PFLD().to(device)
    gaze_pfld.load_state_dict(checkpoint['gaze_pfld'])

    gaze_pfld.eval()

    gaze_pfld = gaze_pfld.to(device)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

    img = cv2.imread('5.png')
    height, width = img.shape[:2]

    input = cv2.resize(img, (160,112))
    input = transform(input).unsqueeze(0).to(device)
    landmarks, gaze = gaze_pfld(input)

    pre_landmark = landmarks[0]
    #print(pre_landmark.shape)
    pre_landmark = pre_landmark.cpu().detach().numpy().reshape(
        -1, 2) * [width, height]

    gaze = gaze.cpu().detach().numpy()[0]

    c_pos = pre_landmark[-1,:]

    cv2.line(img, tuple(c_pos.astype(int)), tuple(c_pos.astype(int)+(gaze*400).astype(int)), (0,255,0), 1)
 
    for (x, y) in pre_landmark.astype(np.int32):
        cv2.circle(img, (x, y), 1, (0, 0, 255))

    cv2.imshow('gaze estimation', img)
    cv2.imwrite('gaze.jpg', img)
    cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path',
                        default="./checkpoint/snapshot/checkpoint_epoch_1.pth.tar",
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
