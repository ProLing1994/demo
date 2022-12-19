import cv2


def sobel(img):

    # Sobel
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    x = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
    y = cv2.convertScaleAbs(y)
    img = cv2.addWeighted(x, 0.5, y, 0.5, 0)

    return img