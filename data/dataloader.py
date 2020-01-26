import torch
import torch.utils.data as data
import numpy as np
import cv2

class Transform():
    
    def __init__(self, low, high, delta):
        self.low = low
        self.high = high
        self.delta = delta
    
    def AdditiveNoise(self, img, gt):
        if np.random.rand() > 0.5:
            return img, gt
        noise = np.random.uniform(-self.delta, self.delta)
        img = img + noise
        return img, gt
    
    def RandomContrast(self, img, gt):
        if np.random.rand() > 0.5:
            return img, gt
        scale = np.random.uniform(self.low, self.high)
        img = img * scale
        return img, gt
    
    def RandomCrop(self, img, gt):
        if np.random.rand() > 0.5:
            return img, gt
        xmin = torch.min(gt[:, 1])
        ymin = torch.min(gt[:, 2])
        xmax = torch.max(gt[:, 3])
        ymax = torch.max(gt[:, 4])
        dx = [np.random.uniform(0, xmin), np.random.uniform(xmax, 1)]
        dy = [np.random.uniform(0, ymin), np.random.uniform(ymax, 1)]
        x = np.array(dx) * 300
        y = np.array(dy) * 300
        img = img[int(y[0]): int(y[1]), int(x[0]): int(x[1]), :]
        img = cv2.resize(img, (300, 300))
        gt[:, 1] = (gt[:, 1]-dx[0]) / (dx[1]-dx[0])
        gt[:, 2] = (gt[:, 2]-dy[0]) / (dy[1]-dy[0])
        gt[:, 3] = (gt[:, 3]-dx[0]) / (dx[1]-dx[0])
        gt[:, 4] = (gt[:, 4]-dy[0]) / (dy[1]-dy[0])
        return img, gt
    
    def Compose(self, img, gt):
        img, gt = self.RandomCrop(img, gt)
        img, gt = self.RandomContrast(img, gt)
        img, gt = self.AdditiveNoise(img, gt)
        return img, gt

def DataAllocate(batch):
    imgs = []
    bboxs = []
    paths = []
    for img, bbox, path in batch:
        imgs.append(img)
        bboxs.append(bbox)
        paths.append(path)
    imgs = torch.stack(imgs)
    return imgs, bboxs, paths
    
class TigerData(data.Dataset):
    
    def __init__(self, anno, training, low=0.8, high=1.2, delta=16.0):
        self.transform = Transform(low, high, delta)
        self.training = training
        if self.training:
            self.anno = anno['train']
            self.list = list(anno['train'].keys())
        else:
            self.anno = anno['test']
            self.list = list(anno['test'].keys())

    def __len__(self):
        return len(self.list)        
    
    def __getitem__(self, idx):
        index = self.list[idx]
        img = self.anno[index]['path']
        bbox = self.anno[index]['anno'].clone()
        img = cv2.imread(img)
        img = cv2.resize(img, (300, 300))
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        if self.training:
            img, bbox = self.transform.Compose(img, bbox)
        #img = img / 255.0
        img = img - np.array([0.485, 0.456, 0.406])*255.0
        img = img / np.array([0.229, 0.224, 0.225])
        img = img.transpose([2, 0, 1])
        if self.training and (torch.rand(1) > 0.5):
            img = img[:, :, ::-1]
            tmp = 1.0 - bbox[:, 1]
            bbox[:, 1] = 1.0 - bbox[:, 3]
            bbox[:, 3] = tmp
        return torch.FloatTensor(img.copy()), bbox.cuda(), self.anno[index]['path'].split('/')[-1]
