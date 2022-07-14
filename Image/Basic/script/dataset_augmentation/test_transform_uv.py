import os
import numpy as np
import cv2


def RGBToNV12(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # y = (0.299 * r + 0.587 * g + 0.114 * b)
    # u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
    # v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
    
    # y = 0.257 * r  + 0.504 * g + 0.098 * b + 16
    # u = (-0.148 * r - 0.291 * g + 0.439 * b + 128)[::2, ::2]
    # v = (0.439 * r - 0.368 * g - 0.071 * b + 128)[::2, ::2]

    y = (66.0/256.0 * r + 129.0/256.0 * g + 25.0/256.0 * b + 128.0/256.0) + 16
    u = ((-38.0/256.0 * r - 74.0/256.0 * g + 112.0/256.0 * b + 128.0/256.0) + 128)[::2, ::2]
    v = ((112.0/256.0 * r - 94.0/256.0 * g - 18.0/256.0 * b + 128.0/256.0) + 128)[::2, ::2]

    uv = np.zeros(shape=(u.shape[0], u.shape[1]*2))
    for i in range(0, u.shape[0]):
        for j in range(0, u.shape[1]):
            uv[i, 2*j] = u[i, j]
            uv[i, 2*j+1] = v[i, j]

    yuv = np.vstack((y, uv))
    return yuv.astype(np.uint8)


def RGBToNV21(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # y = (0.299 * r + 0.587 * g + 0.114 * b)
    # u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
    # v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
    
    # y = 0.257 * r  + 0.504 * g + 0.098 * b + 16
    # u = (-0.148 * r - 0.291 * g + 0.439 * b + 128)[::2, ::2]
    # v = (0.439 * r - 0.368 * g - 0.071 * b + 128)[::2, ::2]

    y = (66.0/256.0 * r + 129.0/256.0 * g + 25.0/256.0 * b + 128.0/256.0) + 16
    u = ((-38.0/256.0 * r - 74.0/256.0 * g + 112.0/256.0 * b + 128.0/256.0) + 128)[::2, ::2]
    v = ((112.0/256.0 * r - 94.0/256.0 * g - 18.0/256.0 * b + 128.0/256.0) + 128)[::2, ::2]
    
    uv = np.zeros(shape=(u.shape[0], u.shape[1]*2))
    for i in range(0, u.shape[0]):
        for j in range(0, u.shape[1]):
            uv[i, 2*j] = v[i, j]
            uv[i, 2*j+1] = u[i, j]

    yuv = np.vstack((y, uv))
    return yuv.astype(np.uint8)


if __name__ == "__main__":

    input_img_path = "/mnt/huanyuan2/data/image/ZG_BMX_detection/banmaxian_test_image/2M_RongHeng_far/0000000000000000-220604-070000-080000-000003000220-sn00180.jpg"
    output_img_path = "/mnt/huanyuan2/data/image/ZG_BMX_detection/banmaxian_test_image/test/"

    img = cv2.imread(input_img_path)
    
    # TO RGB
    img = img[:, :, ::-1]

    # TO YUV，NV12
    NV12 = RGBToNV12(img)
    # NV12 = RGBToNV21(img)
    
    # Write YUV
    NV12.tofile(output_img_path + 'test.yuv')

    # # TO RGB
    # # COLOR_YUV2RGB_NV12: int
    # COLOR_YUV2RGB_NV21: int
    # img = cv2.cvtColor(NV12, cv2.COLOR_YUV2RGB_NV12)
    img = cv2.cvtColor(NV12, cv2.COLOR_YUV2RGB_NV21)
    print(img.shape)

    # TO BGR
    img = img[:, :, ::-1]

    cv2.imwrite(output_img_path + 'test.jpg', img)