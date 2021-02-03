import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import json

dataset_dir = ""
root_dir = ""
img_dir = os.path.join(dataset_dir, "images")
TYPE = "json"  # json

raw_dir = os.path.join(dataset_dir, TYPE)

class2id = { }


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    x = 0 if x < 0 else x
    x = 1 if x > 1 else x
    y = 0 if y < 0 else y
    y = 1 if y > 1 else y
    return (x, y, w, h)


def convert_annotation_xml(image_id):
    in_file_path = os.path.join(raw_dir, image_id + ".xml")
    if os.path.exists(in_file_path):
        in_file = open(in_file_path)
        out_file = open(os.path.join(root_dir, "labels", image_id + ".txt"), 'w')
        try:
            tree = ET.parse(in_file)
        except Exception as e:
            print(in_file_path)
            return False
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in class2id.keys() or int(difficult) == 1:
                continue
            cls_id = class2id[cls]
            xmlbox = obj.find('bndbox')
            b = (
                float(
                    xmlbox.find('xmin').text), float(
                    xmlbox.find('xmax').text), float(
                    xmlbox.find('ymin').text), float(
                        xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            st = str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
            out_file.write(st)
        return True
    else:
        print(in_file_path, "does not exist !!!")
        return False


def convert_annotation_json(image_id):
    in_file_path = os.path.join(raw_dir, image_id + ".json")
    if os.path.exists(in_file_path):
        in_file = open(in_file_path)
        out_file = open(os.path.join(root_dir, "labels", image_id + ".txt"), 'w')
        try:
            root = json.load(in_file)
        except Exception as e:
            print(in_file_path)
            return False
        w = int(root["imageWidth"])
        h = int(root["imageHeight"])

        for obj in root["shapes"]:
            cls = obj["label"]
            if cls not in class2id.keys():
                continue
            cls_id = class2id[cls]
            box = obj["points"]
            b = (
                float(
                    box[0][0]), float(
                    box[1][0]), float(
                    box[0][1]), float(
                    box[1][1]))
            bb = convert((w, h), b)
            st = str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
            out_file.write(st)
        return True
    else:
        print(in_file_path, "does not exist !!!")
        return False


if TYPE == "xml":
    convert_annotation_func = convert_annotation_xml
if TYPE == "json":
    convert_annotation_func = convert_annotation_json

image_ids = os.listdir(img_dir)
list_file = open(os.path.join(root_dir, "total_list.txt"), 'w')
for image_id in image_ids:
    image_id = image_id[:-4]
    if convert_annotation_func(image_id):
        list_file.write(os.path.join(root_dir, "images", '%s.jpg\n'%(image_id)))
list_file.close()
