import os
import cv2
from tqdm import tqdm
from net import SegNeti
from random import seed, shuffle


def get_filelist(path, typefile=['.png']):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    Filelist = [i.replace('\\', '/') for i in Filelist]
    Filelist = [i for i in Filelist if os.path.splitext(i)[-1] in typefile]
    return Filelist


if __name__ == '__main__':
    
    model_dir = "/mnt/huanyuan/model/image/lanenet/Seg_epoch202.pth"
    save_dir = '/mnt/huanyuan/model/image/lanenet/output'
        
    # schoolbus
    seg = SegNeti(model_dir, (128,128), num_class=3, mode='lanenet')
    seg.device = 'cuda:0'

    # # run videos
    # avi_dir = '/mnt/huanyuan2/data/image/ZD_SafeIsland/test_video/轻量级数据/B/'
    # seg.inference_video(avi_dir, 'save', save_dir, wpath = model_dir)

    # run imgs 
    img_dir = '/mnt/huanyuan2/data/image/ZD_SafeIsland/test_video/轻量级数据/B_jpg/0000000000000000-211024-104625-104641-00020B000011/' 
    img_list = os.listdir(img_dir)
    
    for i in tqdm(range(len(img_list))):
        img_path = os.path.join(img_dir, img_list[i])
        img = cv2.imread(img_path)
        res = seg.deal(img)
        cv2.imwrite('{}/res_{}.jpg'.format(save_dir, str(i)), res)