import numpy as np
import sys
import os
import cv2

caffe_root = '/home/huanyuan/code/caffe_ssd-ssd/'
sys.path.insert(0, caffe_root + 'python')

import caffe

# caffe.set_device(0)
# caffe.set_mode_gpu()

net_file = "/mnt/huanyuan/model_final/image_model/zd_ssd_rfb_wmr/ssd_mbv2_2class/model/ssd_mobilenetv2_fpn.prototxt"
caffe_model = "/mnt/huanyuan/model_final/image_model/zd_ssd_rfb_wmr/ssd_mbv2_2class/model/ssd_mobilenetv2_0421.caffemodel"

net = caffe.Net(net_file, caffe_model, caffe.TEST)

# CLASSES = ('__background__', 'obj')
# CLASSES = ('__background__', 'car')   ## car,  license_plate
CLASSES = ('car', 'license_plate')   ## car,  license_plate


def preprocess(src):
    img = cv2.resize(src, (300, 300)).astype(np.float32)

    rgb_mean = np.array((104, 117, 123), dtype=np.int)

    img -= rgb_mean

    img = img.astype(np.float32)

    # img = img * 0.007875
    return img


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0, 0, :, 1]
    conf = out['detection_out'][0, 0, :, 2]
    return (box.astype(np.int32), conf, cls)


def detect(origimg, p_idx):

    img = preprocess(origimg)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    print('out:', out)
    # 'detection_out': array([[[[0.        , 1.        , 0.99971765, 0.71088743, 0.14606021,
    #       0.8703594 , 0.31698763],
    #      [0.        , 1.        , 0.9899526 , 0.0364562 , 0.36921954,
    #       0.12086702, 0.5494108 ]]]], dtype=float32)}
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
        if conf[i] > 0.4:
            p1 = (int(box[i][0]), int(box[i][1]))
            p2 = (int(box[i][2]), int(box[i][3]))
            cv2.rectangle(origimg, p1, p2, (0, 255, 0), 2)
            # p3 = (int(p1[0]/2), int(p1[1]/2))
            p3 = (int((box[i][0] + box[i][2])/2), int((box[i][1] + box[i][3])/2))
            title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
            # title = "%.2f C" % (conf[i])
            cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

    save_file = os.path.join("/mnt/huanyuan2/data/image/ZD_anpr/test_video/ZD_DUBAI/avi文件/5M_白天_侧向_0615/jpg/test.jpg")

    cv2.imwrite(save_file, origimg)

    # cv2.imshow("test", origimg)
    # cv2.waitKey(0)
    return True


if __name__ == '__main__':
    print('This is main ....')
    img_path = "/mnt/huanyuan2/data/image/ZD_anpr/test_video/ZD_DUBAI/avi文件/5M_白天_侧向_0615/jpg/0000000000000003-220615-061710-061722-00050E226151/0000000000000003-220615-061710-061722-00050E226151_00200.jpg"
    opencv_image = cv2.imread(img_path)
    detect(opencv_image, 'test_caffe')