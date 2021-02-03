#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2020/6/29
"""
import os
import cv2
import numpy as np

def clip(x, min, max):
    if isinstance(x, str):
        x = int(x)
    if x < min:
        x = min
    if x >= max:
        x = max -1
    return x

def transfrom(txt_path, image_dir, label_dir):
    datas = []
    for line in open(txt_path, "r").readlines():
        data = line.split()
        image_path = data[0]
        image_name = os.path.basename(image_path)
        bboxes = data[1:]

        # 创建图片软连接
        src = image_path
        dest = os.path.join(image_dir, image_name)
        if not os.path.exists(dest):
            os.symlink(src, dest)
        datas.append(dest)
        # labels
        label_path = os.path.join(label_dir, image_name[:-3] + "txt")
        image = cv2.imread(dest)
        H, W, C = image.shape
        with open(label_path, "w") as writer:
            st = ""
            for bbox in bboxes:
                xmin, ymin, xmax, ymax, cid = bbox.split(",")
                xmin = clip(xmin, 0, W)
                ymin = clip(ymin, 0, H)
                xmax = clip(xmax, 0, W)
                ymax = clip(ymax, 0, H)

                width = xmax - xmin + 1
                height = ymax - ymin + 1
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                cx /= W
                cy /= H
                width /= W
                height /= H
                st += str(cid) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height) + "\n"
            writer.write(st)
    return datas

def creat_list(data, path):
    with open(path, "w") as writer:
        for d in data:
            writer.write(d)
            writer.write("\n")

train_txt_path = ""
val_txt_path = ""

img_dir = ""
label_dir = ""
data_list = ""

data = transfrom(train_txt_path, img_dir, label_dir)
creat_list(data, data_list)
