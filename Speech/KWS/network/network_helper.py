import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_features(feature_map, output_dir):
    # mkdir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(feature_map.shape[1]):
        fig = plt.figure(figsize=(20, 4))
        plt.axis('off')

        img = feature_map[0, i, :, :].T
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255     # float 在 [0，1] 之间，转换成 0-255
        img = img.astype(np.uint8)                                # 转成 unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)            # 生成 heat map
        img = img[:, :, ::-1]                                     # 注意 cv2（BGR）和 matplotlib(RGB) 通道是相反的

        plt.imshow(img)
        print("{}/{}".format(i, feature_map.shape[1]))
        fig.savefig(os.path.join(output_dir, "channel_{}.png".format(i)), dpi=600)
        fig.clf()
        plt.close()