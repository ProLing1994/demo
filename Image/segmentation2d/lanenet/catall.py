# cat 在一起
from utils.utils import buildDir
import cv2
import numpy as np
import os
from tqdm import tqdm

sdir = 'tmp/cat'
buildDir(sdir)

imgdir = 'tmp/crossformer'
imglist = os.listdir(imgdir)
imglist = [i for i in imglist if i.endswith('.jpg')]

for i in tqdm(range(len(imglist))):
    p0 = 'tmp/base/{}'.format(imglist[i])
    p1 = 'tmp/crossformer/{}'.format(imglist[i])
    p2 = 'tmp/segformer/{}'.format(imglist[i])
    p3 = 'tmp/swin/{}'.format(imglist[i])
    
    img0 = cv2.imread(p0)
    img1 = cv2.imread(p1)
    img2 = cv2.imread(p2)
    img3 = cv2.imread(p3)
    
    h, w, _ = img1.shape
    wrate = 0.65
    hrate = 0.85
    size = 1.5
    linw = 5
    color = (0,0,255)
    cv2.putText(img0, 'Base', (int(w*wrate), int(h*hrate)), cv2.FONT_ITALIC, size, color, linw)
    cv2.putText(img1, 'CrossFormer', (int(w*wrate), int(h*hrate)), cv2.FONT_ITALIC, size, color, linw)
    cv2.putText(img2, 'SegFormer', (int(w*wrate), int(h*hrate)), cv2.FONT_ITALIC, size, color, linw)
    cv2.putText(img3, 'Swin', (int(w*wrate), int(h*hrate)), cv2.FONT_ITALIC, size, color, linw)
    
    blank = np.ones((h, 10, 3), dtype=np.uint8) * 255
    cat = np.concatenate([img0, blank, img1, blank, img2, blank, img3], 1)
    
    cv2.imwrite('{}/{}'.format(sdir, imglist[i]), cat)
    # cv2.imshow('', cat)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
