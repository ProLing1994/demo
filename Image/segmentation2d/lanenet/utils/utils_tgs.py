
import os
import cv2
import csv
import datetime
import numpy as np
from tqdm import tqdm
from shutil import copy


def rleToMask(rleString, height, width):
    rows,cols = height,width
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def csv2dict(csvpath):
    csvfile = {}
    with open(csvpath, 'r+') as f:
        for info in f:
            info = info.strip()
            _id, _lb = info.split(',')
            if _id != 'id':
                csvfile[_id] = _lb
    return csvfile

if __name__ == '__main__':
    path = 'D:\Project\Learning\segmentation\Seg_pytorch\\results\inputs\\tgs'
    imgs = os.listdir(path)
    csvfile = csv2dict('D:\Project\Learning\DATA\TGS\submission/Unet_resnet_version_11.csv')

    # # view csv on img
    # for i in range(len(imgs)):
    #     imgname = imgs[i]
    #     imgcode = imgname.split('.')[0]
    #     aimgpath = '{}/{}'.format(path, imgname)
    #     img = cv2.imread(aimgpath)

    #     shape = csvfile[imgcode]
    
    #     if len(shape)>0:
    #         n = len(shape)
    #         mask = rleToMask(shape, 101, 101)
    #         mask = mask.reshape(101, 101, 1)
    #         mask = mask.repeat(3, 2)
            
    #         mask_img = cv2.addWeighted(img, 1, mask, 0.5, 0)
            
    #         cv2.imshow('', mask_img)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    
        
    # new submit
    now = datetime.datetime.now()
    y,m,d,a,b = now.year, now.month, now.day, now.hour, now.minute
    originpath = 'D:/Project/Learning/DATA/TGS/sample_submission.csv'
    targetpath = 'D:/Project/Learning/DATA/TGS/submission/tgs_{}_{}_{}_{}_{}.csv'.format(y,m,d,a,b)
    copy(originpath, targetpath)

    # init csv
    csvfile = []
    with open(targetpath, 'r+') as f:
        for i, v in enumerate(f):
            if i>0:
                v = v.strip()
                _id = v.split(',')[0]
                csvfile.append([_id])

    # mask 2 csv
    path = 'results/outputs/tgs_fpn16_ce_RFBResBlock/0'
    res = os.listdir(path)
    for i in tqdm(range(len(csvfile))):
        img_id = csvfile[i][0]
        img_path = '{}/{}'.format(path, res[i])
        label = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if np.sum(label) != 0:
            rle = mask2rle(label)
        else:
            rle = ''
        csvfile[i].append(rle)
        
    head = ['id', 'rle_mask']
    with open(targetpath, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(head)
        f_csv.writerows(csvfile)
        

    
    
    



