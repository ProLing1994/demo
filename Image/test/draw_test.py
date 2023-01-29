import cv2
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # img = np.zeros((720, 1280, 3))
    img = np.zeros((1080, 1920, 3))
    print(img.shape)

    # cv2.circle(img, (952, 1035), 5, [0, 0, 255], 5)
    # cv2.circle(img, (1875, 7), 5, [255, 0, 0], 5)
    # cv2.circle(img, (1117, 7), 5, [0, 255, 0], 5)

    # cv2.circle(img, (868, 533), 5, [255, 255, 255], 5)
    # cv2.circle(img, (1079, 575), 5, [255, 255, 255], 5)
    # cv2.circle(img, (1009, 1024), 5, [255, 255, 255], 5)
    # cv2.circle(img, (647, 888), 5, [255, 255, 255], 5)

    # cv2.imwrite('/mnt/huanyuan/model/image/test/test_tanker_10_59_59.jpg', img.astype(np.uint8))
    
    cv2.line(img, (500,469), (1920,110), (255,0,0), 3) 
    
    cv2.rectangle(img, (0,543), (70,543+130), (0,255,0), -1)
    cv2.rectangle(img, (422,205), (422+186,205+115), (0,255,0), -1)
    cv2.rectangle(img, (153,356), (153+109,356+130), (0,255,0), -1)
    cv2.rectangle(img, (64,475), (64+83,475+104), (0,255,0), -1)
    cv2.rectangle(img, (1107,111), (1107+192,111+79), (0,255,0), -1)

    # cv2.circle(img, (153, 356), 5, [0, 0, 255], 5)
    # cv2.circle(img, (153+109, 356+130), 5, [0, 0, 255], 5)

    # cv2.circle(img, (0, 543), 5, [0, 0, 255], 5)
    # cv2.circle(img, (70, 543+130), 5, [0, 0, 255], 5)

    cv2.imwrite('/mnt/huanyuan/model/image/test/test_zd.jpg', img.astype(np.uint8))
