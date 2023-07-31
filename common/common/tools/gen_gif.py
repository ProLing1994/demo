import numpy as np
import imageio
import os
from PIL import Image
import re

def draw_gif(folder_dir, image_jpgs, gif_size, gif_path, duration):
    """
    draw gif graph according to image_jpgs
    :param folder_drr:  image dir
    :param image_jpgs: image list
    :param gif_path: outpute path
    :param duration: the time interval between two frames, in seconds
    :return:
    """
    images = []
    for jpg in image_jpgs:
        fpath = os.path.join(folder_dir, jpg)
        im = Image.open(fpath)
        im = im.resize([gif_size[0], gif_size[1]])
        images.append(np.array(im))
    imageio.mimsave(gif_path, images, duration=duration)

if __name__ == '__main__':
    input_path = "/mnt/huanyuan/temp/test/gif/h/"
    gif_name = '/mnt/huanyuan/temp/test/gif/h.gif'
    gif_size = (680, 512)
    duration = 0.15

    image_list = os.listdir(input_path)
    image_list.sort()

    draw_gif(input_path, image_list, gif_size, gif_name, duration)