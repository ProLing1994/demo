# make a datalist.txt for training
# For image segmentation only
# You need to put different class of pictures in different folders

# datadir
#  |--src
#  |   |--img1
#  |   |--img2
#  |   |--...
#  |--label
#  |   |--png1
#  |   |--png2
#  |   |--...
#  |--datalist.txt (result of makeDataList)

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import copy

def tryload(imgpath):
    # tryload imgs to avoid `nonetype`
    img = cv2.imread(imgpath)
    try:
        img.shape
        # cv2.imwrite(imgpath, img)
        return 1
    except:
        return 0

def copyimg(datadir):
    # To avoid warnings when reading pictures
    imglist = []
    with open('{}/datalist.txt'.format(datadir), 'r+') as f:
        for i in f:
            imglist.append(i)

    for i in tqdm(range(len(imglist))):
        path, _ = imglist[i].strip().split(', ')
        path = '{}/{}'.format(datadir, path)
        img = cv2.imread(path)
        cv2.imwrite(path, img)

def makeDataList(datadir):
    datalist_path = '{}/datalist.txt'.format(datadir)
    src_list = os.listdir('{}/src'.format(datadir))
    lbl_list = os.listdir('{}/label'.format(datadir))
    
    src_tmp = [i.split('.')[0] for i in src_list]
    lbl_tmp = [i.split('.')[0] for i in lbl_list]
    assert src_tmp == lbl_tmp
    
    N = len(src_list)
    infolist = []
    for i in tqdm(range(N)):
        rsrcpath = 'src/{}'.format(src_list[i])
        rlblpath = 'label/{}'.format(lbl_list[i])
        asrcpath = '{}/src/{}'.format(datadir, src_list[i])
        alblpath = '{}/label/{}'.format(datadir, lbl_list[i])
        if tryload(asrcpath):
            info = '{}, {}\n'.format(rsrcpath, rlblpath)
            infolist.append(info)
    with open(datalist_path, 'w+') as f:
        for i in infolist:
            f.write(i)
    return datalist_path

def sumPixels(nclass, shape, datadir):
    infopath = '{}/weightlist.txt'.format(datadir)
    h, w = shape
    count = [0]*nclass
    masks = np.ones((nclass, h, w))
    for i in range(nclass):
        masks[i, ...] *= i
    
    datalist = []
    with open('{}/datalist.txt'.format(datadir), 'r+') as f:
        for i in f:
            _, labelpath = i.strip().split()
            datalist.append(labelpath)
    
    for i in tqdm(range(len(datalist))):
        rpath = datalist[i]
        apath = '{}/{}'.format(datadir, rpath)
        label = cv2.imread(apath, cv2.IMREAD_UNCHANGED)
        h_l, w_l = label.shape
        if h!=h_l or w!=w_l:
            raise RuntimeError('size mismatch')
        for j in range(nclass):
            c = np.sum(label == masks[j])
            count[j] += c
    
    avg = np.mean(count)
    count = [avg/i for i in count]
    info_list = ['{}, {}\n'.format(i, v) for i, v in enumerate(count)]
    
    with open(infopath, 'w+') as f:
        for i in info_list:
            f.write(i)
    return count

def read_weight_list(datadir):
    wlist = []
    with open('{}/weightlist.txt'.format(datadir), 'r+') as f:
        for i in f:
            wlist.append(float(i.strip().split(', ')[1]))
    return wlist

def rgb2mask(mask, colormap):
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for i, label in enumerate(colormap):
        label_mask[np.where(mask == label)[:2]] = i
    label_mask = label_mask.astype(int)
    return label_mask

def trans_fig(srcdir, tgtdir, colormap):
    imglist = os.listdir(srcdir)
    colormap = np.array(colormap)
    for i in tqdm(range(len(imglist))):
        apath = '{}/{}'.format(srcdir, imglist[i])
        origin_img = cv2.imread(apath)
        trans_img = rgb2mask(origin_img, colormap)
        newpath = '{}/{}'.format(tgtdir, imglist[i])
        cv2.imwrite(newpath, trans_img)
    return 0

if __name__ == '__main__':
    pass

    # trans_fig('D:\Project\Learning\DATA\TGS\\train\label_view', 
    #           'D:\Project\Learning\DATA\TGS\\train\label', 
    #           [[0,0,0], [255,255,255]])
    
    # datadir = 'D:\Project\Learning\DATA\BDCI\\train'
    datadir = 'D:/Project/Learning/DATA/bdd10k/train'
    
    print('making datalist...')
    datalist_path = makeDataList(datadir)
    print('counting pixels...')
    count = sumPixels(20, (720, 1280), datadir)
    
    wlist = read_weight_list(datadir)
    print(wlist)





