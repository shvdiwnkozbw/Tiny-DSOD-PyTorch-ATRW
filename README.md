# Tiny-DSOD-PyTorch-ATRW

This is a PyTorch implementation of Tiny-DSOD for ATRW Dataset Amur-Tiger Detection.

It is slightly different from the original caffe version [Tiny-DSOD](https://github.com/lyxok1/Tiny-DSOD), which achieves 51.1 mAP on ATRW Dataset.

This PyTorch version is easier to deploy, and is now for reference on Amur-Tiger detection.

It currently achives 47.1 mAP, we are still working on improving the performance.

## Requirements

- Python 3.5 
- PyTorch 0.4.1
- opencv 3.4.1
- pycocotools

## Running Procedure

#### Data preparation

- Download ATRW Dataset from [CVWC](https://cvwc2019.github.io/) 
- Split training set and validation set following [Tiny-DSOD](https://github.com/lyxok1/Tiny-DSOD)
- Generate VOC-form annotation running ```./data/data.py```

#### Training

```
main.py [-h]
optinal arguments:
[--dataset] name of dataset
[--epochs] totoal training epochs
[--train-batch] batchsize for training
[--val-batch] batchsize for validation
[--lr] learning rate in training
[--schedule] decrease learning rate at these epochs
[-tp] number of training epochs between test
[--iter-size] forward-backward times in training
[--checkpoint] checkpoint path to save model
[-e] only to do evaluation
```

```
python3 main.py --train-batch 32 --iter-size 4 --gpu 0
```

#### Validation

```
python3 main.py -e --resume checkpoint/Tiger/model.pth.tar --val-batch 2
python3 utils/eval.py
```

Running ```main.py``` in evaluation mode generates detection results in ```./utils/res.json``` in form of COCO.

Running ```./utils/eval.py``` outputs mAP on ATRW Dataset.

### Performance

[Tiny-DSOD caffe version](https://github.com/lyxok1/Tiny-DSOD) achieves 51.1 mAP on ATRW Dataset.

This PyTorch version currently achives 47.1 mAP.

We are still working on improving the performance.
