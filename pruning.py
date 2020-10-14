#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2020/9/7
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
import copy
import matplotlib.pyplot as plt
from models.yolo import Model


def load_model(cfg="models/mobile-yolo5l_voc.yaml", weights="./outputs/mvoc/weights/best_mvoc.pt"):
    restor_num = 0
    ommit_num = 0
    model = Model(cfg).to(device)
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    dic = {}
    for k, v in ckpt['model'].float().state_dict().items():
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
            dic[k] = v
            restor_num += 1
        else:
            ommit_num += 1

    print("Build model from", cfg)
    print("Resotre weight from", weights)
    print("Restore %d vars, ommit %d vars" % (restor_num, ommit_num))

    ckpt['model'] = dic
    model.load_state_dict(ckpt['model'], strict=False)
    del ckpt

    model.float()
    model.model[-1].export = True
    return model

def bn_analyze(prunable_modules, save_path=None):
    bn_val = []
    max_val = []
    for layer_to_prune in prunable_modules:
        # select a layer
        weight = layer_to_prune.weight.data.detach().cpu().numpy()
        max_val.append(max(weight))
        bn_val.extend(weight)
    bn_val = np.abs(bn_val)
    max_val = np.abs(max_val)
    bn_val = sorted(bn_val)
    max_val = sorted(max_val)
    plt.hist(bn_val, bins=101, align="mid", log=True, range=(0, 1.0))
    if save_path is not None:
        if os.path.isfile(save_path):
            os.remove(save_path)
        plt.savefig(save_path)
    return bn_val, max_val

def channel_prune(ori_model, example_inputs, output_transform, pruned_prob=0.3, thres=None):
    model = copy.deepcopy(ori_model)
    model.cpu().eval()

    prunable_module_type = (nn.BatchNorm2d)

    ignore_idx = [230, 260, 290]

    prunable_modules = []
    for i, m in enumerate(model.modules()):
        if i in ignore_idx:
            continue
        if isinstance(m, prunable_module_type):
            prunable_modules.append(m)
    ori_size = tp.utils.count_params(model)
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs,
                                               output_transform=output_transform)
    bn_val, max_val = bn_analyze(prunable_modules, "render_img/before_pruning.jpg")
    if thres is None:
        thres_pos = int(pruned_prob * len(bn_val))
        thres_pos = min(thres_pos, len(bn_val)-1)
        thres_pos = max(thres_pos, 0)
        thres = bn_val[thres_pos]
    print("Min val is %f, Max val is %f, Thres is %f" % (bn_val[0], bn_val[-1], thres))

    for layer_to_prune in prunable_modules:
        # select a layer
        weight = layer_to_prune.weight.data.detach().cpu().numpy()
        if isinstance(layer_to_prune, nn.Conv2d):
            if layer_to_prune.groups > 1:
                prune_fn = tp.prune_group_conv
            else:
                prune_fn = tp.prune_conv
            L1_norm = np.sum(np.abs(weight), axis=(1, 2, 3))
        elif isinstance(layer_to_prune, nn.BatchNorm2d):
            prune_fn = tp.prune_batchnorm
            L1_norm = np.abs(weight)

        pos = np.array([i for i in range(len(L1_norm))])
        pruned_idx_mask = L1_norm < thres
        prun_index = pos[pruned_idx_mask].tolist()
        if len(prun_index) == len(L1_norm):
            del prun_index[np.argmax(L1_norm)]

        plan = DG.get_pruning_plan(layer_to_prune, prune_fn, prun_index)
        plan.exec()

    bn_analyze(prunable_modules, "render_img/after_pruning.jpg")

    with torch.no_grad():

        out = model(example_inputs)
        if output_transform:
            out = output_transform(out)
        print("  Params: %s => %s" % (ori_size, tp.utils.count_params(model)))
        if isinstance(out, (list, tuple)):
            for o in out:
                print("  Output: ", o.shape)
        else:
            print("  Output: ", out.shape)
        print("------------------------------------------------------\n")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="models/mobile-yolo5s_voc.yaml", type=str, help='*.cfg path')
    parser.add_argument('--weights', default="outputs/smvocs/weights/best_smvocs.pt", type=str, help='*.data path')
    parser.add_argument('--save-dir', default="outputs/smvocs/weights", type=str, help='*.data path')
    parser.add_argument('-p', '--prob', default=0.5, type=float, help='pruning prob')
    parser.add_argument('-t', '--thres', default=0, type=float, help='pruning thres')
    opt = parser.parse_args()

    cfg = opt.cfg
    weights = opt.weights
    save_dir = opt.save_dir

    device = torch.device('cpu')
    model = load_model(cfg, weights)

    example_inputs = torch.zeros((1, 3, 64, 64), dtype=torch.float32).to()
    output_transform = None
    # for prob in [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    if opt.thres != 0:
        thres = opt.thres
        prob = "p.auto"
    else:
        thres = None
        prob = opt.prob

    pruned_model = channel_prune(model, example_inputs=example_inputs,
                                 output_transform=output_transform, pruned_prob=prob, thres=thres)
    pruned_model.model[-1].export = False
    save_path = os.path.join(save_dir, "pruned_"+str(prob).split(".")[-1] + ".pt")
    torch.save({"model": pruned_model.module if hasattr(pruned_model, 'module') else pruned_model}, save_path)

