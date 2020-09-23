import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
voc_dir = "/data/dataset/voc"
root_dir = "/work/yolov5-pruning-distillation/dataset/voc"
sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    x = 0 if x < 0 else x
    x = 1 if x > 1 else x
    y = 0 if y < 0 else y
    y = 1 if y > 1 else y
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open(os.path.join(voc_dir, 'VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id)))
    out_file = open(os.path.join(root_dir, "labels", 'VOC%s/%s.txt'%(year, image_id)), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')



for year, image_set in sets:
    if not os.path.exists(os.path.join(root_dir, "labels", 'VOC%s/'%(year))):
        os.makedirs(os.path.join(root_dir, "labels", 'VOC%s/'%(year)))
    image_ids = open(os.path.join(voc_dir, 'VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set))).read().strip().split()
    list_file = open(os.path.join(root_dir, '%s_%s.txt'%(year, image_set)), 'w')
    for image_id in image_ids:
        list_file.write(os.path.join(root_dir, "images", 'VOC%s/%s.jpg\n'%(year, image_id)))
        convert_annotation(year, image_id)
    list_file.close()

# os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
# os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")
os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > voc_train.txt")
os.system("cat 2007_test.txt > voc_test.txt")