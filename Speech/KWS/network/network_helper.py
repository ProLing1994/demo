import cv2
import matplotlib.pyplot as plt
import numpy as np

def draw_features(width, height, feature_map, output_path):
    fig = plt.figure(figsize=(32, 32))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = feature_map[0, i, :, :].T
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255     # float 在 [0，1] 之间，转换成 0-255
        img = img.astype(np.uint8)                                # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)            # 生成 heat map
        img = img[:, :, ::-1]                                     # 注意 cv2（BGR）和 matplotlib(RGB) 通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(output_path, dpi=100)
    fig.clf()
    plt.close()