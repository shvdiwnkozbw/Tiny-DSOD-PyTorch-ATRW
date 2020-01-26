#import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
#import skimage.io as io
#import pylab

annType = ['segm','bbox','keypoints']
annType = 'bbox'      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'


annFile = 'val.json'
cocoGt=COCO(annFile)

cocoDt=cocoGt.loadRes('res.json')

imgIds=sorted(cocoGt.getImgIds())
#imgIds=imgIds[0:100]
#imgId = imgIds[np.random.randint(100)]

cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
