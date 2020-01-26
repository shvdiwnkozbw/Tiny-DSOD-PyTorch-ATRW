import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

def ComputeIOU(priors, gts):
    IOU = torch.zeros(priors.shape[0], gts.shape[0]).cuda()
    for i in range(gts.shape[0]):
        gt = gts[i]
        inter_x = torch.max((torch.min(priors[:, 2], gt[2])-torch.max(priors[:, 0], gt[0])), torch.cuda.FloatTensor([0.0]))
        inter_y = torch.max((torch.min(priors[:, 3], gt[3])-torch.max(priors[:, 1], gt[1])), torch.cuda.FloatTensor([0.0]))
        union = (priors[:, 2]-priors[:, 0])*(priors[:, 3]-priors[:, 1]) + (gt[2]-gt[0])*(gt[3]-gt[1])
        iou = inter_x * inter_y / (union-inter_x*inter_y)
        IOU[:, i] = iou
    return IOU

def encode_bbox(prior, gt, variance):
    dx = 0.5*(gt[0]+gt[2]-prior[:, 0]-prior[:, 2]) / (prior[:, 2]-prior[:, 0])
    dy = 0.5*(gt[1]+gt[3]-prior[:, 1]-prior[:, 3]) / (prior[:, 3]-prior[:, 1])
    dw = torch.log((gt[2]-gt[0])/(prior[:, 2]-prior[:, 0]))
    dh = torch.log((gt[3]-gt[1])/(prior[:, 3]-prior[:, 1]))
    if len(variance) == 2:
        vc = variance[0]
        vl = variance[1]
    else:
        vc = variance[0]
        vl = variance[0]
    return torch.stack([dx/vc, dy/vc, dw/vl, dh/vl], 1)

def decode_bbox(prior, offset, variance):
    xc = 0.5*(prior[:, 0]+prior[:, 2])
    yc = 0.5*(prior[:, 1]+prior[:, 3])
    w = prior[:, 2] - prior[:, 0]
    h = prior[:, 3] - prior[:, 1]
    if len(variance) == 2:
        vc = variance[0]
        vl = variance[1]
    else:
        vc = variance[0]
        vl = variance[0]
    dx = vc * w * offset[:, 0]
    dy = vc * h * offset[:, 1]
    dw = torch.exp(offset[:, 2]*vl)
    dh = torch.exp(offset[:, 3]*vl)
    new = torch.zeros_like(offset)
    new[:, 0] = xc + dx - 0.5*w*dw
    new[:, 1] = yc + dy - 0.5*h*dh
    new[:, 2] = xc + dx + 0.5*w*dw
    new[:, 3] = yc + dy + 0.5*h*dh
    return new

def nms(score, bbox, threshold, top_k=400):
    assert score.shape[0] == bbox.shape[0]
    keep = torch.zeros_like(score)
    keep[score>0.01] = 1
    for i in range(bbox.shape[0]-1):
        if torch.sum(keep[:i]) >= top_k:
            keep[i:] = 0
            break
        if torch.sum(keep[i:]) == 0:
            break
        if keep[i] == 0:
            continue
        IOU = ComputeIOU(bbox[(i+1):, :], bbox[i:(i+1), :])
        mask = (IOU[:, 0] > threshold)
        dropidx = torch.arange(i+1, score.shape[0])[mask]
        keep[dropidx] = 0
    keep = keep.type(torch.uint8)
    return score[keep], bbox[keep, :]

def normalize(feat, scale=20.0):
    norm = torch.sum(feat*feat, 1, keepdim=True)
    feat = feat / torch.sqrt(norm)
    feat = feat * scale
    return feat

class Normalize(nn.Module):
    
    def __init__(self, in_channel, across_spaticl, channel_shared):
        super(Normalize, self).__init__()
        self.spatial = across_spaticl
        self.channel = channel_shared
        if channel_shared:
            self.scale = Parameter(torch.FloatTensor(1))
        else:
            self.scale = Parameter(torch.FloatTensor(1, in_channel, 1, 1))
        init.constant(self.scale, 20.0)
        
    def forward(self, feat):
        norm = torch.sum(feat*feat, 1, keepdim=True)
        if self.spatial:
            norm = torch.mean(torch.mean(norm, -1, keepdim=True), -2, keepdim=True)
        feat = feat / torch.sqrt(norm)
        feat = self.scale * feat
        return feat

class DDB_B(nn.Module):
    
    def __init__(self, n, g):
        super(DDB_B, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(n, g, (1, 1), stride=1, padding=0, bias=False),
                nn.BatchNorm2d(g, eps=0.001, momentum=0.001),
                nn.Conv2d(g, g, (3, 3), stride=1, padding=1, groups=g, bias=False),
                nn.BatchNorm2d(g, eps=0.001, momentum=0.001),
                nn.ReLU(True)
        )
        init.xavier_uniform_(self.conv[0].weight)
        init.xavier_uniform_(self.conv[2].weight)
    
    def forward(self, feat):
        inter = self.conv(feat)
        feat = torch.cat([feat, inter], dim=1)
        return feat

class Dense(nn.Module):
    
    def __init__(self, layers, n, g):
        super(Dense, self).__init__()
        self.dense = nn.ModuleList()
        for idx in range(layers):
            self.dense.append(DDB_B(n+idx*g, g))
    
    def forward(self, feat):
        for layer in self.dense:
            feat = layer(feat)
        return feat

class Transition(nn.Module):
    
    def __init__(self, in_channel, out_channel, pooling):
        super(Transition, self).__init__()
        self.pooling = pooling
        self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, (1, 1), stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channel, eps=0.001, momentum=0.001),
                nn.ReLU(True)
        )
        init.xavier_uniform_(self.conv[0].weight)
        if self.pooling:
            self.pool = nn.MaxPool2d((3, 3), stride=2, padding=1)
        
    def forward(self, feat):
        feat = self.conv(feat)
        if self.pooling:
            pool = self.pool(feat)
            return feat, pool
        return feat

class DownSample(nn.Module):
    
    def __init__(self, in_channel=128, multi=True):
        super(DownSample, self).__init__()
        self.multi = multi
        self.path_1 = nn.Sequential(
                nn.MaxPool2d((3, 3), stride=2, padding=1),
                nn.Conv2d(in_channel, 64, (1, 1), stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64, eps=0.001, momentum=0.001),
                nn.ReLU(True)
        )
        init.xavier_uniform_(self.path_1[1].weight)
        if self.multi:
            self.path_2 = nn.Sequential(
                    nn.Conv2d(in_channel, 64, (1, 1), stride=1, padding=0, bias=False),
                    nn.Conv2d(64, 64, (3, 3), stride=2, padding=1, groups=64, bias=False),
                    nn.BatchNorm2d(64, eps=0.001, momentum=0.001),
                    nn.ReLU()
            )
            init.xavier_uniform_(self.path_2[0].weight)
            init.xavier_uniform_(self.path_2[1].weight)
        
    def  forward(self, feat):
        if self.multi:
            feat = torch.cat([self.path_1(feat), self.path_2(feat)], 1)
        else:
            feat = self.path_1(feat)
        return feat
    
class UpSample(nn.Module):
    
    def __init__(self, size, in_channel=128):
        super(UpSample, self).__init__()
#        self.upsample = nn.UpsamplingBilinear2d(size=size)
        self.size = size
        self.conv = nn.Conv2d(in_channel, 128, (3, 3), stride=1, padding=1, groups=in_channel, bias=False)
        init.xavier_uniform_(self.conv.weight)
    
    def forward(self, feat):
#        feat = self.upsample(feat)
        feat = F.interpolate(feat, size=self.size, mode='bilinear')
        feat = self.conv(feat)
        return feat

class FPN(nn.Module):
    
    def __init__(self):
        super(FPN, self).__init__()
        self.down_1 = DownSample(multi=False)
        self.down_2 = DownSample()
        self.down_3 = DownSample()
        self.down_4 = DownSample()
        self.down_5 = DownSample()
        self.up_1 = UpSample(3)
        self.up_2 = UpSample(5)
        self.up_3 = UpSample(10)
        self.up_4 = UpSample(19)
        self.up_5 = UpSample(38)
        self.conv = nn.Conv2d(128, 128, (1, 1), stride=1, padding=0, bias=False)
        self.transform = nn.ReLU(True)
        init.xavier_uniform_(self.conv.weight)
    
    def forward(self, feat_1, feat_2):
        feat = [[]] * 11
        feat[0] = feat_1
        feat[1] = torch.cat([feat_2, self.down_1(feat_1)], 1)
        feat[2] = self.down_2(feat[1])
        feat[3] = self.down_3(feat[2])
        feat[4] = self.down_4(feat[3])
        feat[5] = self.down_5(feat[4])
        feat[6] = self.transform(self.up_1(feat[5]) + feat[4])
        feat[7] = self.transform(self.up_2(feat[6]) + feat[3])
        feat[8] = self.transform(self.up_3(feat[7]) + feat[2])
        feat[9] = self.transform(self.up_4(feat[8]) + feat[1])
#        feat[10] = self.transform(self.up_5(feat[9]) + feat[0])
        feat[10] = self.transform(self.conv(self.up_5(feat[9])) + feat[0])
        return feat

class Predictor(nn.Module):
    
    def __init__(self, num_class):
        super(Predictor, self).__init__()
        self.num_class = num_class
        self.boxes = [4, 6, 6, 6, 4, 4]
        self.normalize = nn.ModuleList()
#        self.normalize = Normalize(128, False, False)
        self.locPredict = nn.ModuleList()
        self.clsPredict = nn.ModuleList()
        for i in range(len(self.boxes)-1, -1, -1):
            locPredictor = nn.Sequential(
                    nn.Conv2d(128, 4*self.boxes[i], (1, 1), stride=1, padding=0, bias=False),
                    nn.Conv2d(4*self.boxes[i], 4*self.boxes[i], (3, 3), stride=1, padding=1, groups=4*self.boxes[i], bias=False),
                    nn.BatchNorm2d(4*self.boxes[i], eps=0.001, momentum=0.001)
            )
            init.normal_(locPredictor[0].weight, mean=0, std=0.01)
            self.locPredict.append(locPredictor)
            clsPredictor = nn.Sequential(
                    nn.Conv2d(128, self.num_class*self.boxes[i], (1, 1), stride=1, padding=0, bias=False),
                    nn.Conv2d(self.num_class*self.boxes[i], self.num_class*self.boxes[i], (3, 3), stride=1, padding=1, groups=self.num_class*self.boxes[i], bias=False),
                    nn.BatchNorm2d(self.num_class*self.boxes[i], eps=0.001, momentum=0.001)
            )
            init.normal_(clsPredictor[0].weight, mean=0, std=0.1)
            self.clsPredict.append(clsPredictor)
            self.normalize.append(Normalize(128, False, False))

            
    def forward(self, feat):
        assert len(feat) == len(self.boxes)
        loc = [[]] * len(self.boxes)
        conf = [[]] * len(self.boxes)
        for i in range(len(feat)):
            feat[i] = self.normalize[i](feat[i])
            loc[i] = self.locPredict[i](feat[i])
            conf[i] = self.clsPredict[i](feat[i])
            loc[i] = loc[i].reshape([loc[i].shape[0], 4, -1])
            conf[i] = conf[i].reshape([conf[i].shape[0], self.num_class, -1])
        loc.reverse()
        conf.reverse()
        loc = torch.cat(loc, dim=-1)
        conf = torch.cat(conf, dim=-1)
        return loc, conf

class PriorBox(nn.Module):
    
    def __init__(self):
        super(PriorBox, self).__init__()
        self.orig = 300
        self.size = [38, 19, 10, 5, 3, 2]
        self.min_size = [30.0, 60.0, 111.0, 162.0, 213.0, 264.0]
        self.max_size = [60.0, 111.0, 162.0, 213.0, 264.0, 315.0]
        self.aspect_ratio = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        
    def forward(self):
        priors = [[]] * len(self.size)
        for i in range(len(priors)):
            anchors = torch.zeros(self.size[i], self.size[i], 2+2*len(self.aspect_ratio[i]), 4)
            for y in range(self.size[i]):
                center_y = (y+0.5) / self.size[i]
                for x in range(self.size[i]):
                    center_x = (x+0.5) / self.size[i]
                    w = self.min_size[i] / self.orig
                    anchors[y, x, 0, 0] = center_x - 0.5*w
                    anchors[y, x, 0, 1] = center_y - 0.5*w
                    anchors[y, x, 0, 2] = center_x + 0.5*w
                    anchors[y, x, 0, 3] = center_y + 0.5*w
                    w = np.sqrt(self.min_size[i]*self.max_size[i]) / self.orig
                    anchors[y, x, 1, 0] = center_x - 0.5*w
                    anchors[y, x, 1, 1] = center_y - 0.5*w
                    anchors[y, x, 1, 2] = center_x + 0.5*w
                    anchors[y, x, 1, 3] = center_y + 0.5*w
                    for j in range(len(self.aspect_ratio[i])):
                        w = self.min_size[i] * np.sqrt(self.aspect_ratio[i][j]) / self.orig
                        h = self.min_size[i] / (np.sqrt(self.aspect_ratio[i][j])*self.orig)
                        anchors[y, x, 2+2*j, 0] = center_x - 0.5*w
                        anchors[y, x, 2+2*j, 1] = center_y - 0.5*h
                        anchors[y, x, 2+2*j, 2] = center_x + 0.5*w
                        anchors[y, x, 2+2*j, 3] = center_y + 0.5*h
                        anchors[y, x, 3+2*j, 0] = center_x - 0.5*h
                        anchors[y, x, 3+2*j, 1] = center_y - 0.5*w
                        anchors[y, x, 3+2*j, 2] = center_x + 0.5*h
                        anchors[y, x, 3+2*j, 3] = center_y + 0.5*w
            priors[i] = anchors.permute(3, 2, 0, 1).contiguous()
            priors[i] = priors[i].reshape([4, -1]).permute(1, 0).contiguous()
        priors = torch.cat(priors, 0)
        priors[priors<0] = 0.0
        priors[priors>1] = 1.0
        return priors

class Framework(nn.Module):
    
    def __init__(self, num_class):
        super(Framework, self).__init__()
        self.stem = nn.Sequential(
                nn.Conv2d(3, 64, (3, 3), stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64, eps=0.001, momentum=0.001),
                nn.ReLU(True),
                nn.Conv2d(64, 64, (1, 1), stride=1, padding=0, bias=False),
                nn.Conv2d(64, 64, (3, 3), stride=1, padding=1, groups=64, bias=False),
                nn.BatchNorm2d(64, eps=0.001, momentum=0.001),
                nn.ReLU(True),
                nn.Conv2d(64, 128, (1, 1), stride=1, padding=0, bias=False),
                nn.Conv2d(128, 128, (3, 3), stride=1, padding=1, groups=128, bias=False),
                nn.BatchNorm2d(128, eps=0.001, momentum=0.001),
                nn.ReLU(True),
                nn.MaxPool2d((3, 3), stride=2, padding=1)
        )
        init.xavier_uniform_(self.stem[0].weight)
        init.xavier_uniform_(self.stem[3].weight)
        init.xavier_uniform_(self.stem[4].weight)
        init.xavier_uniform_(self.stem[7].weight)
        init.xavier_uniform_(self.stem[8].weight)
        self.dense_0 = Dense(4, 128, 32)
        self.transition_0 = Transition(256, 128, True)
        self.dense_1 = Dense(6, 128, 48)
        self.transition_1 = Transition(416, 128, True)
        self.dense_2 = Dense(6, 128, 64)
        self.transition_2 = Transition(512, 256, False)
        self.dense_3 = Dense(6, 256, 80)
        self.transition_3 = Transition(736, 64, False)
        self.fpn = FPN()
        self.predictor = Predictor(num_class)
        
    def forward(self, feat):
        feat = self.stem(feat)
        feat = self.dense_0(feat)
        _, feat = self.transition_0(feat)
        feat = self.dense_1(feat)
        first, feat = self.transition_1(feat)
        feat = self.dense_2(feat)
        feat = self.transition_2(feat)
        feat = self.dense_3(feat)
        feat = self.transition_3(feat)
        feat = self.fpn(first, feat)
        loc, conf = self.predictor(feat[5:])
        return loc, conf

class GTGenerator():
    
    def __init__(self, threshold, variance):
        super(GTGenerator, self).__init__()
        self.threshold = threshold
        self.variance = variance
    
    def forward(self, priors, gts):
        batchsize = len(gts)
        targets = - torch.ones(batchsize, priors.shape[0], 5).cuda()
        for i in range(batchsize):
            label = gts[i][:, 0]
            box = gts[i][:, 1:]
            iou = ComputeIOU(priors, box)
            max_iou, index = torch.max(iou, 1)
            mask = (max_iou < self.threshold)
            index[mask] = -1
            gt_idx = torch.argmax(iou, 0)
            index[gt_idx] = torch.arange(box.shape[0]).cuda()
            for idx in range(box.shape[0]):
                mask = (index == idx)
                if torch.sum(mask) == 0:
                    continue
                targets[i, mask, 0] = label[idx]
                targets[i, mask, 1:] = encode_bbox(priors[mask, :], box[idx, :], self.variance)
        return targets

class MultiBoxLoss(nn.Module):
    
    def __init__(self, weight, neg_ratio):
        super(MultiBoxLoss, self).__init__()
        self.weight = weight
        self.neg_ratio = neg_ratio
        self.celoss = nn.CrossEntropyLoss(size_average=False, reduce=False)
        self.l1loss = nn.SmoothL1Loss(size_average=False)        
    
    def forward(self, conf, loc, target):
        assert target.shape[0] == conf.shape[0] == loc.shape[0]
#        print(loc.shape, conf.shape)
        assert target.shape[1] == conf.shape[-1] == loc.shape[-1]
        target.requires_grad = False
        num_class = conf.shape[1]
        target = target.reshape([-1, 5])
        conf = conf.permute(0, 2, 1).contiguous().reshape([-1, num_class])
        loc = loc.permute(0, 2, 1).contiguous().reshape([-1, 4])
        pos_idx = (target[:, 0] > -1)
        neg_idx = (target[:, 0] == -1)
        closs = self.celoss(conf, (target[:, 0]+1).type(torch.cuda.LongTensor))
        lloss = self.l1loss(loc[pos_idx, :], target[pos_idx, 1:])
        negloss = closs[neg_idx]
        pos_num = torch.sum(pos_idx).item()
        neg_num = torch.sum(neg_idx).item()
        max_neg = min(neg_num, pos_num*self.neg_ratio)
        negloss, _ = torch.sort(negloss, descending=True)
        closs = torch.sum(closs[pos_idx]) + torch.sum(negloss[:int(max_neg)])
        loss = torch.sum(closs) / float(pos_num+max_neg) + self.weight * torch.sum(lloss) / float(pos_num)
        return loss

class Detector(nn.Module):
    
    def __init__(self, threshold, variance):
        super(Detector, self).__init__()
        self.threshold = threshold
        self.variance = variance
        
    def forward(self, conf, loc, prior):
        assert conf.shape[0] == loc.shape[0]
        assert prior.shape[0] == loc.shape[-1] == conf.shape[-1]
        batchsize = conf.shape[0]
        scores = [[]] * batchsize
        boxes = [[]] * batchsize
        loc = loc.permute(0, 2, 1).contiguous()
        conf = conf.permute(0, 2, 1).contiguous()
        conf = torch.softmax(conf, -1)
        conf = conf[:, :, 1:]
        bbox = torch.zeros_like(loc)
        for i in range(batchsize):
            bbox[i, :, :] = decode_bbox(prior, loc[i], self.variance)
        for i in range(conf.shape[-1]):
            score, ids = torch.sort(conf[:, :, i], dim=1, descending=True)
#            print('\n', score)
            for j in range(batchsize):
                box = bbox[j, ids[j, :], :]
                scores[j], boxes[j] = nms(score[j, :], box, self.threshold)
#        print(scores[0].shape)
        return scores, boxes
