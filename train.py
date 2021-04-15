#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import logging
from pathlib import Path
import time
import os
import cv2
import numpy as np
import torch

from torch.utils import data
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset.datasets import EyesDataset
from models.pfld import PFLDInference, AuxiliaryNet, Gaze_PFLD
from pfld.loss import PFLDLoss
from pfld.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, gaze_pfld, criterion, optimizer, epoch):
    losses = AverageMeter()

    gaze_loss, loss = None, None
    for i_batch, (img, landmark_gt, gaze_gt) in enumerate(train_loader):
        image = img.to(device)
        landmark_gt = landmark_gt.to(device)
        gaze_gt = gaze_gt.to(device)

        gaze_pfld = gaze_pfld.to(device)
        landmarks, gaze = gaze_pfld(image)

        if 1:
            c_img = np.uint8(image.detach().cpu().numpy()[0][0] * 255)
            c_img = cv2.cvtColor(c_img, cv2.COLOR_GRAY2BGR)
            c_lad = landmark_gt.detach().cpu().numpy()[0]
            c_ladpre = landmarks.detach().cpu().numpy()[0].reshape(-1,2)
            c_gaze = gaze_gt.detach().cpu().numpy()[0]
            c_gazepre = gaze.detach().cpu().numpy()[0]

            c_pos = c_lad[-1,:]
            c_pospre = c_ladpre[-1,:]
            c_pos[0] = c_pos[0]*c_img.shape[1]
            c_pos[1] = c_pos[1]*c_img.shape[0]
            c_pospre[0] = c_pospre[0]*c_img.shape[1]
            c_pospre[1] = c_pospre[1]*c_img.shape[0]

            cv2.line(c_img, tuple(c_pos.astype(int)), tuple(c_pos.astype(int)+(c_gaze*100).astype(int)), (0,255,0), 1)
            cv2.line(c_img, tuple(c_pospre.astype(int)), tuple(c_pospre.astype(int)+(c_gazepre*100).astype(int)), (0,0,255), 1)

            for (x, y) in c_lad:
                color = (0, 255, 0)
                cv2.circle(c_img, (int(round(x*c_img.shape[1])), int(round(y*c_img.shape[0]))), 1, color, -1, lineType=cv2.LINE_AA)
            for (x, y) in c_ladpre:
                color = (0, 0, 255)
                cv2.circle(c_img, (int(round(x*c_img.shape[1])), int(round(y*c_img.shape[0]))), 1, color, -1, lineType=cv2.LINE_AA)                    
            cv2.imshow('c1', c_img)
            cv2.waitKey(1)

        gaze_loss, lad_loss = criterion(landmark_gt.float(),
                                        landmarks, gaze_gt.float(), gaze)
        loss = gaze_loss.float() + lad_loss.float()

        if i_batch % 100 == 0:
            print('Epoch: {} gaze loss: {:.4f}  loss: {:.4f} '.format(epoch, gaze_loss, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
    return loss


def validate(wlfw_val_dataloader, gaze_pfld, criterion):
    gaze_pfld.eval()
    #auxiliarynet.eval()
    losses = []
    with torch.no_grad():
        for img, landmark_gt, gaze_gt in wlfw_val_dataloader:
            img = img.to(device) / 255
            landmark_gt = landmark_gt.to(device)
            gaze_gt = gaze_gt.to(device)
            gaze_pfld = gaze_pfld.to(device)
            _, landmark = gaze_pfld(img)

            landmark_gt = landmark_gt.reshape(-1, 51, 2)
            landmark = landmark.reshape(-1, 51, 2)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            losses.append(loss.cpu().numpy())
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} '.format(np.mean(losses)))
    return np.mean(losses)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    gaze_pfld = Gaze_PFLD().to(device)
    #pfld_backbone = PFLDInference().to(device)
    #auxiliarynet = AuxiliaryNet().to(device)
    criterion = PFLDLoss()
    optimizer = torch.optim.Adam([{'params': gaze_pfld.parameters()}], lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)
    if args.resume:
        checkpoint = torch.load(args.resume)
        gaze_pfld.load_state_dict(checkpoint["gaze_pfld"])
        #pfld_backbone.load_state_dict(checkpoint["pfld_backbone"])
        args.start_epoch = checkpoint["epoch"]

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    eyedataset = EyesDataset(args.datasets, dataroot=args.dataroot, transforms=transform, input_width=args.input_width, input_height=args.input_height)

    N = len(eyedataset)
    TN = int(N * 0.95)
    VN = N - TN
    train_set, val_set = torch.utils.data.random_split(eyedataset, (TN, VN))
    train_loader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True, num_worker=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.val_batchsize, shuffle=True, num_worker=args.workers)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        train_loss = train(train_loader, gaze_pfld, criterion, optimizer, epoch)
        filename = os.path.join(str(args.snapshot),
                                "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint(
            {
                'epoch': epoch,
                'gaze_pfld': gaze_pfld.state_dict(),
                #'auxiliarynet': auxiliarynet.state_dict()
            }, filename)

        val_loss = validate(val_loader, gaze_pfld,
                            criterion)

        scheduler.step(val_loss)
        writer.add_scalars('data/loss', {
            'val loss': val_loss,
            'train loss': train_loss
        }, epoch)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='eyegaze')
    # general
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD

    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=500, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',
                        default='./checkpoint/snapshot/',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--log_file',
                        default="./checkpoint/train.logs",
                        type=str)
    parser.add_argument('--tensorboard',
                        default="./checkpoint/tensorboard",
                        type=str)
    parser.add_argument(
        '--resume',
        default='./checkpoint/snapshot/checkpoint_epoch_6.pth.tar',
        type=str,
        metavar='PATH')

    # --dataset
    parser.add_argument('--dataroot',
                        default='/opt/sda5/BL01_Data/EyeGaze_Data',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--datasets', type=str, default='E', help='datasets flages.')
    parser.add_argument('--input_width', type=int, default=160, help='input size.')
    parser.add_argument('--input_height', type=int, default=112, help='input size.')
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=256, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
