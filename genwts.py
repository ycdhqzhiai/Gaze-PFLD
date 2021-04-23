import os
import argparse
import torch
import struct

from models.pfld import Gaze_PFLD

def parse_args():
    parser = argparse.ArgumentParser(description='tensorrt wts')
    parser.add_argument('--weights', type=str, default='checkpoint/snapshot/checkpoint_epoch_387.pth.tar')
    parser.add_argument('--input_width', type=int, default=160, help='input size.')
    parser.add_argument('--input_height', type=int, default=112, help='input size.')
    args = parser.parse_args()
    return args

def main(args):
    print('cuda device count: ', torch.cuda.device_count())
    device = 'cuda:0'
    checkpoint = torch.load(args.weights, map_location=device)
    net = Gaze_PFLD().to(device)
    net.load_state_dict(checkpoint['gaze_pfld'])

    net.eval()
    print('model: ', net)

    input = torch.ones(1, 3, args.input_height, args.input_width).to(device)
    lad, gaze = net(input)

    print(len(net.state_dict().keys()))
    wts_file = os.path.split(args.weights)[1].replace('.pth.tar', '.wts')
    with open(wts_file, 'w') as f:
        f.write("{}\n".format(len(net.state_dict().keys())))
        for k, v in net.state_dict().items():
            print('key: ', k)
            print('value: ', v.shape)
            val = v.reshape(-1).cpu().numpy()
            f.write("{} {}".format(k, len(val)))
            for vv in val:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)