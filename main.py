import torch
import torch.utils.data as data
import torch.optim as optim
from dataloader import TigerData, DataAllocate
from framework import Framework, MultiBoxLoss, GTGenerator, PriorBox, Detector
from logger import AverageMeter, Logger
import pickle
import os
import argparse
import shutil
import time
import cv2
import numpy as np
import json
from progress.bar import Bar

def parse_args():
    parser = argparse.ArgumentParser(description='Tiger Detection')

    parser.add_argument('-d', '--dataset', default='Tiger', type=str)
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')

    parser.add_argument('--epochs', default=1200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=2, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--val-batch', default=1, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[400, 800],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--warm-up', dest='wp', default=100, type=int, 
                        help='warm up learning rate epoch')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--val-per-epoch','-tp', dest='tp', default=2000, type=int,
                        help='number of training epoches between test (default: 30)')
    parser.add_argument('--iter-size', '-is', dest='its', default=4, type=int,
                        help='the forward-backward times within each iteration')

    parser.add_argument('--pretrained', '-pre', dest='pre', default=False, type=bool, 
                        help='whether to use pretrained model')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    return args

args = parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_gpu = torch.cuda.is_available() and int(args.gpu_id) >= 0

with open('det.pkl', 'rb') as f:
    info = pickle.load(f, encoding='bytes')

def main():

    start_epoch = args.start_epoch
    max_mAP = 0.0
    
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    
    print('Preparing Dataset %s' % args.dataset)
    
    trainset = TigerData(info, training=True)
    valset = TigerData(info, training=False)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
                                  collate_fn=DataAllocate)
    valloader = data.DataLoader(valset, batch_size=args.val_batch, shuffle=False, num_workers=args.workers, 
                                collate_fn=DataAllocate)
    
    framework = Framework(2)
    multibox_loss = MultiBoxLoss(1.0, 3.0)
    gt_generator = GTGenerator(0.5, [0.1, 0.2])
    priors = PriorBox()().cuda()
    detector = Detector(0.5, [0.1, 0.2])
    
    if use_gpu:
        framework = framework.cuda()
        multibox_loss = multibox_loss.cuda()
        detector = detector.cuda()
        
    optimizer = optim.SGD(framework.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    
#    optimizer = optim.Adam(framework.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    args.checkpoint = os.path.join(args.checkpoint, args.dataset)
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)
    
    if args.resume:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='Tiger Detection', resume=True)
        checkpoint = torch.load(args.resume)
        framework.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='Tiger Detection', resume=False)
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss'])
    
    if args.evaluate:
        test(valloader, framework, priors, detector, start_epoch, use_gpu)
        logger.close()
        return
    
    for epoch in range(start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)        
#        if epoch == 0:
#            warm_up_lr(optimizer, True)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, args.lr))
        
        loss = train(trainloader, priors, framework, multibox_loss, gt_generator, optimizer, epoch, use_gpu)
        
        if (epoch+1) % args.tp == 0:
            test(valloader, framework, priors, detector, epoch, use_gpu)    
            logger.append([int(epoch+1), args.lr, loss])
        else:
            logger.append([int(epoch+1), args.lr, loss])
        
        filename = str(epoch) if (epoch+1)%40 == 0 else 'model' 
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': framework.state_dict(),
            'loss': loss,
            'optimizer': optimizer.state_dict()
        }, epoch + 1, checkpoint=args.checkpoint, filename=filename)
    
    logger.close()

def train(trainloader, priors, model, multibox, gt, optimizer, epoch, use_gpu):
    
    model.train()
    multibox.train()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    end = time.time()
    
    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()
    
    for batch_idx, (imgs, boxes, paths) in enumerate(trainloader):
                
#        if  (batch_idx == args.wp) and (epoch == 0):
#            warm_up_lr(optimizer, False)

        data_time.update(time.time() - end)
        if use_gpu:
            imgs = imgs.cuda()

        data_time.update(time.time() - end)
        loc, conf = model(imgs)
        target = gt.forward(priors, boxes)
        loss = multibox(conf, loc, target)
        
        if loss.item() > 0:
            total_loss.update(loss.item(), 1)
            
        loss /= args.its
        loss.backward()

        if batch_idx % args.its == 0:
            optimizer.step()
            optimizer.zero_grad()
#        visualize(imgs, [priors[target[0, :, 0]>-1]], paths)        
        end = time.time()
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.3f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=total_loss.val
        )
        bar.next()
        
    bar.finish()
    
    return total_loss.avg

def test(valloader, model, priors, detector, epoch, use_gpu):

    model.eval()
    data_time = AverageMeter()
    end = time.time()
    
    bar = Bar('Processing', max=len(valloader))
    result = []
    
    for batch_idx, (imgs, boxes, paths) in enumerate(valloader):
                
        data_time.update(time.time() - end)
        if use_gpu:
            imgs = imgs.cuda()

        data_time.update(time.time() - end)
        loc, conf = model(imgs)
        score, box = detector(conf, loc, priors)
#        print('\n', score)
        rec_result(result, paths, score, box)
#        visualize(imgs, box, paths)
                
        end = time.time()
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s'.format(
            batch=batch_idx + 1,
            size=len(valloader),
            data=data_time.val
        )
        bar.next()
    bar.finish()
    
    with open('res.json', 'w') as f:
        json.dump(result, f)


def rec_result(res, path, score, box):
    assert len(path) == len(score) == len(box)
    for i in range(len(path)):
        image_id = int(path[i].split('.')[0])
        category_id = 1
        for j in range(len(box[i])):
            if score[i][j] < 0.01:
                continue
            bbox = box[i][j].cpu().detach().numpy().copy()
            bbox[bbox<0] = 0.0
            bbox[bbox>1] = 1.0
            conf = score[i][j].cpu().item()
            res.append({'image_id': image_id, 'category_id': category_id, 
                        'bbox': [float(bbox[0]*1920), float(bbox[1]*1080),
                                       float(bbox[2]*1920-bbox[0]*1920), float(bbox[3]*1080-bbox[1]*1080)], 
                        'score': conf})

def visualize(imgs, boxes, paths=[]):
    imgs = imgs.cpu().numpy()
    img = imgs[0].transpose(1, 2, 0)
    #img = img * np.array([0.229, 0.224, 0.225])
    img = img + np.array([0.485, 0.456, 0.406])*255.0
    img[img<0] = 0
    img[img>255] = 255
    #img = img * 255.0
    img = np.uint8(img)
    box = boxes[0].cpu().detach().numpy()
    for i in range(2):
        img = cv2.rectangle(img.copy(), (np.int(box[i, 0]*300), np.int(box[i, 1]*300)), (np.int(box[i, 2]*300), np.int(box[i, 3]*300)), (0, 255, 0), 2)
#    img = cv2.rectangle(img.copy(), (100, 100), (200, 200), (0, 255, 0), 2)
    cv2.imwrite(paths[0], img)

def save_checkpoint(state, epoch, checkpoint='checkpoint', filename='checkpoint'):
    filepath = os.path.join(checkpoint, filename + '.pth.tar')
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def warm_up_lr(optimizer, warm_up):
    if warm_up:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * args.lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

if __name__ == '__main__':
    main()
