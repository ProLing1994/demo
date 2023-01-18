import os
from tqdm import tqdm
from utils.utils import readlist, buildDir
from utils.utils_mask import *
from multiprocessing.dummy import Pool as ThreadPool

datalistpath = 'datalists/det/local/SCAllWash_train.txt'
datalist = readlist(datalistpath)
imglist = [i.split(', ')[0] for i in datalist]

def view(imglist):
    for i in tqdm(range(len(imglist))):
        imgpath = imglist[i]
        imgname = imgpath.split('/')[-1][:-4]
        subdir = imgpath.split('/')[4]
        # jsonpath = 'F:/Projects/Streamax/SegHandWheel/handwheel/{}__{}.json'.format(subdir, imgname)
        jsonpath = 'handwheel/{}__{}.json'.format(subdir, imgname)

        # check json file
        img = cv2.imread(imgpath)
        label = multi_label_mask(img, jsonpath, ['wheel', 'hand'])
        res = multi_label_view(img, label, ['wheel', 'hand'], ((0,255,0), (0,0,255)))
        
        imgspath = 'F:/Projects/Streamax/SegHandWheel/viewautotag/{}__{}.jpg'.format(subdir, imgname)
        cv2.imwrite(imgspath, res)

def split_list(vlist, nworker):
    newlist = []
    n = len(vlist) // nworker
    tail = len(vlist) % nworker
    for i in range(nworker):
        sp = vlist[i*n:(i+1)*n]
        newlist.append(sp)
    for i in range(tail):
        newlist[i].append(vlist[-(i+1)])
    return newlist




Nworker = 8
pool = ThreadPool(Nworker)

mlist = split_list(imglist, Nworker)

results = pool.map(view, mlist)
pool.close()
pool.join()



"""

for i in tqdm(range(len(imglist))):
    imgpath = imglist[i]
    imgname = imgpath.split('/')[-1][:-4]
    subdir = imgpath.split('/')[4]
    # jsonpath = 'F:/Projects/Streamax/SegHandWheel/handwheel/{}__{}.json'.format(subdir, imgname)
    jsonpath = 'handwheel/{}__{}.json'.format(subdir, imgname)

    # check json file
    img = cv2.imread(imgpath)
    label = multi_label_mask(img, jsonpath, ['wheel', 'hand'])
    res = multi_label_view(img, label, ['wheel', 'hand'], ((0,255,0), (0,0,255)))
    
    imgspath = 'F:/Projects/Streamax/SegHandWheel/viewautotag/{}__{}.jpg'.format(subdir, imgname)
    cv2.imwrite(imgspath, res)
    
"""
    