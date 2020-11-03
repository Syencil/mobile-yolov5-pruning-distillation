#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2020/9/23
"""
import torch
import argparse
from thop import profile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    print(opt)

    img = torch.zeros((opt.batch_size, 3, *opt.img_size))

    model = torch.load(opt.weights, map_location=torch.device('cpu'))
    if model.get('model', None) is not None:
        model = model["model"]
    model.float()
    model.eval()
    model.fuse()
    macs, params = profile(model, inputs=(img, ))

    print("Model ===> ", opt.weights)
    print("Macs: ", macs / (1000 ** 3), "G")
    print("Params: ", params / (1000 ** 2), "M")