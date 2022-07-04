import cv2
import os

if __name__ == '__main__':

    image_dir = "/mnt/huanyuan/model_final/image_model/lpr_zg/china/double/image/"
    
    image_list = os.listdir(image_dir)
    image_list.sort()

    for idx in range(len(image_list)):
        image_name = image_list[idx]
        image_path = os.path.join(image_dir, image_name)

        if not image_name.endswith(".jpg"):
            continue
    
        img = cv2.imread(image_path, 0) 
        img = cv2.resize(img, (256, 64))

        cv2.imwrite(image_path, img)
