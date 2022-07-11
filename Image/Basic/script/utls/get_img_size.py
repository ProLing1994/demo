import cv2
import os

if __name__ == '__main__':

    image_dir = "/mnt/huanyuan/model_final/image_model/lpr_zg/china/double/image/"
    
    image_list = os.listdir(image_dir)
    image_list.sort()

    max_height = 0
    max_width = 0

    for idx in range(len(image_list)):
        image_name = image_list[idx]
        image_path = os.path.join(image_dir, image_name)

        if not image_name.endswith(".jpg"):
            continue
    
        img = cv2.imread(image_path, 0) 
        height, width = img.shape[0], img.shape[1]

        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width

    print(max_height, max_width)



