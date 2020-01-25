import os
import torch
import pickle

root_dir = '/media/yuxi/Data/Tiger/detection/yolo/VOC_challenge/labels'
img_dir = '/media/yuxi/Data/Tiger/detection/yolo/VOC_challenge/JPEGImages'
files = os.listdir(root_dir)
train = '/media/yuxi/Data/Tiger/detection/yolo/VOC_challenge/ImageSets/Main/trainval.txt'
test = '/media/yuxi/Data/Tiger/detection/yolo/VOC_challenge/ImageSets/Main/test.txt'

labels = dict()
for file in files:
    with open(os.path.join(root_dir, file), 'r') as f:
        index = int(file.split('.')[0])
        labels[index] = []
        while True:
            bbox = f.readline()
            if len(bbox) == 0:
                break
            cls, xc, yc, w, h = bbox.split(' ')
            cls = float(cls)
            xc = float(xc)
            yc = float(yc)
            w = float(w)
            h = float(h)
            labels[index].append(torch.tensor([cls, xc-0.5*w, yc-0.5*h, xc+0.5*w, yc+0.5*h]))

train_idx = []
test_idx = []

with open(train, 'r') as f:
    while True:
        index = f.readline()
        if len(index) == 0:
            break
        train_idx.append(int(index))

with open(test, 'r') as f:
    while True:
        index = f.readline()
        if len(index) == 0:
            break
        test_idx.append(int(index))

info = dict()
info['train'] = dict()
info['test'] = dict()

for idx in train_idx:
    info['train'][idx] = dict()
    info['train'][idx]['anno'] = torch.stack(labels[idx])
    info['train'][idx]['path'] = os.path.join(img_dir, str(idx).zfill(4)+'.jpg')
for idx in test_idx:
    info['test'][idx] = dict()
    info['test'][idx]['anno'] = torch.stack(labels[idx])
    info['test'][idx]['path'] = os.path.join(img_dir, str(idx).zfill(4)+'.jpg')

with open('det.pkl', 'wb') as f:
    pickle.dump(info, f)