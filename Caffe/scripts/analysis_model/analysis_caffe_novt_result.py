import cv2
import numpy as np 
import os

if __name__ == "__main__":

    # pc resize
    # caffe_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/pc/caffe_res_test_side_00000.xml"
    # novt_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/pc_resize/resize_img_test_side_00000.xml"
    # caffe_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/pc/caffe_res_7118000000000000-221116-225400-225420-01p013000000_00000.xml"
    # novt_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/pc_resize/resize_img_7118000000000000-221116-225400-225420-01p013000000_00000.xml"

    # 板端 cv resize 函数结果
    # caffe_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/pc/caffe_res_test_side_00000.xml"
    # novt_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/novt_cv_resize/test_side_00000.xml"
    # caffe_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/pc/caffe_res_7118000000000000-221116-225400-225420-01p013000000_00000.xml"
    # novt_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/novt_cv_resize/7118000000000000-221116-225400-225420-01p013000000_00000.xml"
    # caffe_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/pc/caffe_res_0927000000000000-221117-073200-073220-01p013000000_00260.xml"
    # novt_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/novt_cv_resize/0927000000000000-221117-073200-073220-01p013000000_00260.xml"
    
    # 板端自带 resize 函数结果
    # caffe_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/pc/caffe_res_test_side_00000.xml"
    # novt_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/novt_resize/test_side_00000.xml"
    # caffe_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/pc/caffe_res_7118000000000000-221116-225400-225420-01p013000000_00000.xml"
    # novt_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/novt_resize/7118000000000000-221116-225400-225420-01p013000000_00000.xml"
    caffe_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/pc/caffe_res_0927000000000000-221117-073200-073220-01p013000000_00260.xml"
    novt_xml_path = "/mnt/huanyuan2/data/image/HY_Tanker/novt_resize/0927000000000000-221117-073200-073220-01p013000000_00260.xml"

    # 加载矩阵
    cv_file = cv2.FileStorage(caffe_xml_path, cv2.FILE_STORAGE_READ)
    cv_net_output = cv_file.getNode("test").mat()
    cv_file.release()

    novt_file = cv2.FileStorage(novt_xml_path, cv2.FILE_STORAGE_READ)
    novt_net_output = novt_file.getNode("test").mat()
    novt_file.release()

    print(cv_net_output.shape)
    print(novt_net_output.shape)

    cv_net_output_float = cv_net_output / 255.0
    novt_net_output_float = novt_net_output / 255.0

    diff_output = cv_net_output_float - novt_net_output_float
    x_list, y_list = np.where(abs(diff_output) > 0.1)
    print(len(x_list))

    # for idx in range(len(x_list)):
    #     print(x_list[idx], y_list[idx], cv_net_output_float[x_list[idx]][y_list[idx]], novt_net_output_float[x_list[idx]][y_list[idx]])

    cv_img = cv_net_output.astype(np.uint8)
    novt_img = novt_net_output.astype(np.uint8)
    output_path = caffe_xml_path[:-4] + '_img.jpg'
    cv2.imwrite(output_path, cv_img)
    output_path = novt_xml_path[:-4] + '_img.jpg'
    cv2.imwrite(output_path, novt_img)