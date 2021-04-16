# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import argparse
import sys
import time
from models.pfld import Gaze_PFLD

import torch
import torch.nn as nn
import models


# def load_model_weight(model, checkpoint):
#     state_dict = checkpoint['model_state_dict']
#     # strip prefix of state_dict
#     if list(state_dict.keys())[0].startswith('module.'):
#         state_dict = {k[7:]: v for k, v in checkpoint['model_state_dict'].items()}

#     model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

#     # check loaded parameters and created model parameters
#     for k in state_dict:
#         if k in model_state_dict:
#             if state_dict[k].shape != model_state_dict[k].shape:
#                 print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
#                     k, model_state_dict[k].shape, state_dict[k].shape))
#                 state_dict[k] = model_state_dict[k]
#         else:
#             print('Drop parameter {}.'.format(k))
#     for k in model_state_dict:
#         if not (k in state_dict):
#             print('No param {}.'.format(k))
#             state_dict[k] = model_state_dict[k]
#     model.load_state_dict(state_dict, strict=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="./checkpoint/snapshot/checkpoint_epoch_1.pth.tar", help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[112, 160], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand

    device = "cpu"
    print("=====> load pytorch checkpoint...")
    checkpoint = torch.load(opt.weights, map_location=torch.device('cpu')) 
    net = Gaze_PFLD().to(device)
    net.load_state_dict(checkpoint['gaze_pfld'])

    img = torch.zeros(1, 3, *opt.img_size).to(device)
    print(img.shape)
    landmarks, gaze = net.forward(img)
    f = opt.weights.replace('.pth.tar', '.onnx')  # filename
    torch.onnx.export(net, img, f,export_params=True, verbose=False, opset_version=12, input_names=['inputs'])
    # # ONNX export
    try:
        import onnx
        from onnxsim import simplify

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pth.tar', '.onnx')  # filename
        torch.onnx.export(net, img, f, verbose=False, opset_version=11, input_names=['images'],
                          output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, f)
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)