
import cv2
import torch
import numpy as np
import torch.utils.data as data

if __name__ == '__main__':
    import sys
    sys.path.append('/lirui/Projects/R151Seg/')
    
from utils.augmentations.aug_cv2 import do_aug
from utils.utils_mask import *
from dataset.prase_target import PlatformTargetParser

class Segdataset(data.Dataset):
    def __init__(self, datalist, CLASSES, target_shape, 
                 aug_rate=0.5, rgb_means=127.5, mode='train', ):
        
        self.datalist = datalist
        self.rgb_means = rgb_means
        self.mode = mode
        self.aug_rate = aug_rate
        self.target_shape = target_shape
        self.CLASSES = CLASSES
        self.seg_classes = CLASSES[1:]
        self.box_classes = []
        self.parser = PlatformTargetParser(self.seg_classes, self.box_classes)

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index):
        info = self.datalist[index]
        src_path, lbl_path = info.split(', ')
        src = cv2.imread(src_path)
        tag = self.parser.parse(lbl_path) # h,w,1
        
        h, w, _ = src.shape
        mask = np.zeros((h, w, 3), dtype=src.dtype)
        for instance in tag["annotations"]:
            if instance["type"] == "polygon":
                clsname = instance['class_name']
                if clsname in self.CLASSES:
                    idx = self.CLASSES.index(clsname)
                    fill_value = int(idx)
                    contours = [np.array(instance['points'])]
                    mask = cv2.drawContours(mask, contours, -1, (fill_value,fill_value,fill_value), cv2.FILLED)
                else:
                    pass
        lbl = mask[..., :1]
        src, lbl = self.process(src, lbl)
        return src, lbl
    
    def process(self, src, lbl):
        if self.mode == 'train':
            # lbl : in hwc, out hwc
            src, lbl = do_aug((src, lbl), self.aug_rate)
        
        if src.shape != self.target_shape:
            src = cv2.resize(src, self.target_shape[::-1])
            lbl = cv2.resize(lbl, self.target_shape[::-1], interpolation=cv2.INTER_NEAREST)
        
        src = src[:, :, ::-1].copy()   # bgr2rgb
        src = src.astype(np.float32)
        src = src - self.rgb_means
        src = src.transpose((2, 0, 1)) # hwc2chw
        return torch.FloatTensor(src), torch.LongTensor(lbl)


def tensor2img(tensor):
    img = tensor + 127.5
    img = img[0].numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.array(img, dtype=np.uint8)
    img = img[..., ::-1]
    img = img.copy()
    return img

if __name__ == '__main__':
    import os
    def get_filelist(path, typefile=['.png']):
        Filelist = []
        for home, dirs, files in os.walk(path):
            for filename in files:
                Filelist.append(os.path.join(home, filename))
        Filelist = [i.replace('\\', '/') for i in Filelist]
        Filelist = [i for i in Filelist if os.path.splitext(i)[-1] in typefile]
        return Filelist
    
    def get_datalist2(datalist):
        trainlist = []
        testlist = []
        for droot in datalist:
            train_datadir = '{}/train'.format(droot)
            test_datadir = '{}/test'.format(droot)

            train_datalist = get_filelist(train_datadir, ['.jpg'])
            test_datalist = get_filelist(test_datadir, ['.jpg'])

            train_id_list = [i.split('/')[-1].split('.jpg')[0] for i in train_datalist]
            test_id_list = [i.split('/')[-1].split('.jpg')[0] for i in test_datalist]
            
            _trainlist = ['{}/{}.jpg, {}/{}.json'.format(train_datadir, i, train_datadir, i) for i in train_id_list]
            _testlist = ['{}/{}.jpg, {}/{}.json'.format(test_datadir, i, test_datadir, i) for i in test_id_list]
            
            trainlist += _trainlist
            testlist += _testlist
        
        return trainlist, testlist

    DATALIST = [
        '/lirui/DATA/SchoolBusSeg/SchoolBusSeg/base', 
    ]
    TRAINLIST, TESTLIST = get_datalist2(DATALIST)
    print(len(TRAINLIST), len(TESTLIST))
    CLASSES = ['_background_', 'rail', 'roadside', 'green_belts', 'person', ]
    colormap = np.array([[0,0,0], [0,255,255], [0,0,255], [0,255,0], [255,0,0], [255,255,0]])
    dataset = Segdataset(TRAINLIST, CLASSES, target_shape=(1080, 1920), aug_rate=0.5, mode='test')
    dataloader = data.DataLoader(dataset, 1, shuffle=True , num_workers=0)

    count = 0
    for batch, (images, labels) in enumerate(dataloader):
        count +=1
        images = tensor2img(images)
        labels = labels.cpu().numpy()
        rgb_mask = mask2rgb(labels[0], colormap)
        res = cv2.addWeighted(images, 1.0, rgb_mask, 0.5, 1.0)
        cv2.imwrite('/lirui/Projects/R151Seg/view/{}.jpg'.format(str(count).zfill(5)), res)
        
        if count==100:
            break
    
    

