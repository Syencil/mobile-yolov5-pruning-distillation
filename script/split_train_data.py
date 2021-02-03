# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2020/6/18
"""
import os
import random


def split(total_txt, train_txt, val_txt,ratio_split=0.1):
    total_list = open(total_txt, "r").readlines()
    random.shuffle(total_list)
    length = int(len(total_list) * ratio_split)
    val_list = total_list[:length]
    train_list = total_list[length:]
    with open(train_txt, "w") as writer:
        writer.writelines(train_list)
    with open(val_txt, "w") as writer:
        writer.writelines(val_list)


if __name__ == '__main__':
    project_name = ""
    root_dir = ""
    total_list = os.path.join(root_dir, project_name, "total_list.txt")
    train_txt = os.path.join(root_dir, project_name, "train_list.txt")
    val_txt = os.path.join(root_dir, project_name, "val_list.txt")
    split(total_list, train_txt, val_txt, 0.1)

