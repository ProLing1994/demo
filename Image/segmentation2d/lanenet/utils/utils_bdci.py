
import cv2
import numpy as np
from glob import glob

size = 256
imgs = glob('D:\Project\Learning\DATA\BDCI\\test/*.png')

def split():
    for k in range(len(imgs)):
        img = cv2.imread('D:\Project\Learning\DATA\BDCI\\test/{}.png'.format(k+1))
        h, w, _ = img.shape

        n_h, n_w = h//size, w//size

        for i in range(n_h+1):
            for j in range(n_w+1):

                sub_img = img[i*256:(i+1)*256, j*256:(j+1)*256, :]
                h, w, _ = sub_img.shape
                
                if h<256 or w<256:
                    tmp = np.zeros((256,256,3))
                    tmp = np.array(tmp, dtype=int)
                    tmp[:h, :w, :] = sub_img
                    sub_img = tmp
                    
                cv2.imwrite('D:\Project\Learning\DATA\BDCI\\test_split/{}_{}_{}.png'.format(k+1, i, j), sub_img)

def cat(path):
    for k in range(len(imgs)):
        img = cv2.imread('D:\Project\Learning\DATA\BDCI\\test/{}.png'.format(k+1))
        h, w, _ = img.shape
        n_h, n_w = h//size, w//size
        tmp = np.zeros(((n_h+1)*256, (n_w+1)*256, 3))
        for i in range(n_h+1):
            for j in range(n_w+1):
                sub_img = cv2.imread('{}/{}_{}_{}.png'.format(path, k+1, i, j))
                tmp[i*256:(i+1)*256, j*256:(j+1)*256, :] = sub_img
        res = tmp[:h, :w, :]
        cv2.imwrite('D:\Project\Learning\DATA\BDCI\\result/{}.png'.format(k+1), res)


if __name__ == '__main__':
    # cat('D:/Project/Learning/segmentation/Seg_pytorch/results/outputs/bdci_seg16_ce_vggwide/2')
    cat('D:/Project/Learning/segmentation/Seg_pytorch/results/outputs/bdci_seg16_ce_invres/2')




