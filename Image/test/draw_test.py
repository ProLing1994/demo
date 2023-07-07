import cv2
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # img = np.zeros((1080, 1920, 3))
    img = cv2.imread('/mnt/huanyuan/temp/test/1.jpg')
    print(img.shape)

    # cv2.line(img, (500,469), (1920,110), (255,0,0), 3) 
    # cv2.rectangle(img, (0,543), (70,543+130), (0,255,0), -1)
    # cv2.circle(img, (153, 356), 5, [0, 0, 255], 5)

    cv2.circle(img, (780, 190), 5, [255, 0, 0], 5)
    cv2.circle(img, (1070, 175), 5, [255, 0, 0], 5)
    cv2.circle(img, (1345, 190), 5, [255, 0, 0], 5)
    cv2.circle(img, (1496, 404), 5, [255, 0, 0], 5)
    cv2.circle(img, (1730, 710), 5, [255, 0, 0], 5)
    cv2.circle(img, (330, 720), 5, [255, 0, 0], 5)
    cv2.circle(img, (613, 387), 5, [255, 0, 0], 5)

    cv2.imwrite('/mnt/huanyuan/temp/test/res.jpg', img.astype(np.uint8))
