import cv2
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # img = np.zeros((720, 1280, 3))
    img = np.zeros((1080, 1920, 3))
    print(img.shape)

    cv2.circle(img, (960, 1035), 5, [0, 0, 255], 5)
    cv2.circle(img, (1875, 7), 5, [255, 0, 0], 5)
    cv2.circle(img, (1117, 7), 5, [0, 255, 0], 5)

    cv2.circle(img, (868, 533), 5, [255, 255, 255], 5)
    cv2.circle(img, (1079, 575), 5, [255, 255, 255], 5)
    cv2.circle(img, (1009, 1024), 5, [255, 255, 255], 5)
    cv2.circle(img, (647, 888), 5, [255, 255, 255], 5)

    cv2.imwrite('/mnt/huanyuan/model/image/test/test_tanker.jpg', img.astype(np.uint8))
