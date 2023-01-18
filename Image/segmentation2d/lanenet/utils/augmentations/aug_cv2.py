import cv2
import numpy as np
import random

def gaussian_blur(img):
    k1 = np.random.choice((1,3))
    k2 = np.random.choice((1,3))
    blurred = cv2.GaussianBlur(img, ksize=(k1, k2), sigmaX=0, sigmaY=0)
    return blurred

def motion_blur(image):
    degree = np.random.randint(2, 5)
    angle  = np.random.randint(0, 180)
    image = np.array(image)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def addnoise(img):
    h, w, _ = img.shape
    noise = np.random.uniform(-8, 8, size=(h, w, 1))
    noi_img = np.clip(img+noise, 0, 255)
    noi_img = np.array(noi_img, dtype=np.uint8)
    return noi_img

def brightness(img):
    alpha = np.random.uniform(0.5, 1.0)
    beta  = np.random.uniform(0.0, 10.0)
    blank = np.zeros(img.shape, img.dtype)
    img = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return img

def gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.expand_dims(gray, 2)
    gray = np.repeat(gray, 3, 2)
    return gray

def colorjitter(img):
    '''
    ### Different Color Jitter ###
    img: image
    cj_type: {b: brightness, s: saturation, c: constast}
    '''
    cjtype = 'bsc'
    idx = np.random.randint(0,3)
    cj_type = cjtype[idx]
    if cj_type == "b":
        value = np.random.randint(-75, 75)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    
    elif cj_type == "s":
        value = np.random.randint(-20, 20)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    
    elif cj_type == "c":
        brightness = np.random.randint(-75, 75)
        contrast = np.random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img

def flip(src, lbl, mode=0):
    # mode0: random
    # mode1: up and down
    # mode2: left and right
    if mode==0:
        mode = np.random.randint(0, 2) + 1
    if mode==1:
        src, lbl = src[::-1, :, :], lbl[::-1, :, :]
    if mode==2:
        src, lbl = src[:, ::-1, :], lbl[:, ::-1, :]
    return src, lbl

def rotate(src, lbl):
    seed = np.random.randint(0, 2)
    if seed:
        src = np.rot90(src)
        lbl = np.rot90(lbl)
    else:
        src = np.rot90(src, k=3)
        lbl = np.rot90(lbl, k=3)
    return src, lbl

def crop(src, lbl, p1=None, p2=None):
    h, w, c = src.shape
    if p1 is None:
        crop_xmin = np.random.randint(0, w//5)
        crop_ymin = np.random.randint(0, h//5)
    else:
        crop_xmin, crop_ymin = p1[0], p1[1]
        # crop_xmin, crop_ymin = p1[0]+np.random.randint(0, w//20), p1[1]+np.random.randint(0, h//20)
    
    if p2 is None:
        crop_xmax = w - np.random.randint(0, w//5)
        crop_ymax = h - np.random.randint(0, h//5)
    else:
        crop_xmax, crop_ymax = p2[0], p2[1]
        # crop_xmax, crop_ymax = p2[0]+np.random.randint(0, w//20), p2[1]+np.random.randint(0, h//20)

    crop_xmin = int(max(0, int(crop_xmin)))
    crop_xmax = int(min(w, int(crop_xmax)))
    crop_ymin = int(max(0, int(crop_ymin)))
    crop_ymax = int(min(h, int(crop_ymax)))

    src = src[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
    lbl = lbl[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
    return src, lbl

def square_crop(src, lbl, p1, p2):
    h, w, c = src.shape
    x1,y1 = p1
    x2,y2 = p2
    
    x_length = x2-x1
    y_length = y2-y1
    max_length = max(x_length, y_length)
    
    x_center = (x1+x2)//2
    y_center = (y1+y2)//2
    
    x1 = x_center - max_length//2
    x2 = x_center + max_length//2
    y1 = y_center - max_length//2
    y2 = y_center + max_length//2
    
    x1 = int(max(0, int(x1)))
    x2 = int(min(w, int(x2)))
    y1 = int(max(0, int(y1)))
    y2 = int(min(h, int(y2)))
    
    src = src[y1:y2, x1:x2, :]
    lbl = lbl[y1:y2, x1:x2, :]
    return src, lbl

def random_crop(src, lbl, p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    h, w = y2-y1, x2-x1
    H, W, _ = src.shape
    
    h_range = h//5
    w_range = w//5
    
    x1 = x1 + np.random.randint(-w_range, w_range*0.25)
    x2 = x2 + np.random.randint(-w_range*0.25, w_range)
    y1 = y1 + np.random.randint(-h_range, h_range*0.25)
    y2 = y2 + np.random.randint(-h_range*0.25, h_range)
    
    x1 = int(max(0, int(x1)))
    x2 = int(min(W, int(x2)))
    y1 = int(max(0, int(y1)))
    y2 = int(min(H, int(y2)))
    
    src = src[y1:y2, x1:x2, :]
    lbl = lbl[y1:y2, x1:x2, :]
    
    return src, lbl

def random_padding(img, lbl):
    h, w, _ = img.shape
    _, _, c = lbl.shape
    scale = np.random.uniform(1.2, 1.5)
    bg_h, bg_w = int(h*scale), int(w*scale)
    rand_px = np.random.randint(0, bg_w-w)
    rand_py = np.random.randint(0, bg_h-h)
    
    src_bgd = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
    src_bgd[rand_py:rand_py+h, rand_px:rand_px+w, :] = img
    
    mask_bgd = np.zeros((bg_h, bg_w, c), dtype=np.uint8)
    mask_bgd[rand_py:rand_py+h, rand_px:rand_px+w, :] = lbl
    return src_bgd, mask_bgd

def do_aug(pair, aug_rate=0.5):
    src, lbl = pair
    if np.random.rand() < aug_rate*0.5:
        src = channel_shuffle(src)
    if np.random.rand() < aug_rate: # 0.0988
        src = gaussian_blur(src)
    if np.random.rand() < aug_rate: # 0.3351
        src = motion_blur(src)
    # if np.random.rand() < aug_rate: # 1.6807
    #     src = addnoise(src)
    # if np.random.rand() < aug_rate: # 0.1117
    #     src = brightness(src)
    if np.random.rand() < aug_rate:
        src = colorjitter(src)
    if np.random.rand() < aug_rate:
        src = gray(src)
    
    if np.random.rand() < aug_rate: # 0.0718
        src, lbl = flip(src, lbl, mode=2)
    # if np.random.rand() < aug_rate: # 0.0718
    #     src, lbl = rotate(src, lbl)
    
    if np.random.rand() < aug_rate:
        if np.random.rand() < 0.5:
            src, lbl = crop(src, lbl)
        else:
            src, lbl = random_padding(src, lbl)
    
    return src, lbl

def channel_shuffle(img):
    img = img.copy()
    idx = [0,1,2]
    random.shuffle(idx)
    return img[..., idx]

def tensor2img(tensor, mask=None):
    img = tensor + 127.5
    img = img[0].numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.array(img, dtype=np.uint8)
    img = img[..., ::-1]
    img = img.copy()
    
    if not mask is None:
        mask = mask[0].numpy()
        mask = np.transpose(mask, (1,2,0))
        mask = np.repeat(mask, 3, 2)
        mask *= 255 
        mask = np.array(mask, dtype=np.uint8)
        img = cv2.addWeighted(img, 0.5, mask, 0.7, 0)
        
    return img




