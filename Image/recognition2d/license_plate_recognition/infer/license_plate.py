import cv2
import numpy as np
import os
import sys
import time
import torch

caffe_root = "/home/huanyuan/code/caffe_ssd/"
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')
import caffe

sys.path.insert(0, '/home/huanyuan/code/demo/')
import Speech.API.Kws_weakup_Asr.impl.asr_decode_beamsearch as Decode_BeamSearch


ocr_labels = ["-","皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
              "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏",
              "陕", "甘", "青", "宁", "新", "警", "学", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '挂']
# ocr_labels = ["-","wan", "hu", "jin", "yu", "ji", "jin", "meng", "liao", "ji", "hei", "su", "zhe",
#               "jing", "min", "gan", "lu", "yu", "e", "xiang", "yue", "gui", "qiong", "chuan", "gui", "yun", "zang",
#               "shan", "gan", "qing", "ning", "xin", "jing", "xue", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
#               'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
#               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'gua']

def preprocess(src):
    img = cv2.resize(src, (256, 64))
    img = img / 255.0
    return img


def license_palte_model_init_caffe(caffe_prototxt, caffe_model):
    # init
    net = caffe.Net(caffe_prototxt, caffe_model, caffe.TEST)

    return net


def license_palte_beamsearch_init():
    # init
    asr_beamsearch = Decode_BeamSearch.BeamSearch(ocr_labels, beam_size=5)
    lm = None

    return (asr_beamsearch, lm)


def license_palte_crnn_recognition_caffe(net, img):

    def greedy_decode( probs, blank_id = 0 ):
        
        prob_idxs = np.argmax( probs, axis=1 )
        
        first_pass = []
        first_pass_score = []
        for idx in range(len(prob_idxs)):
            prob_idx = prob_idxs[idx]
            if len(first_pass) == 0 or prob_idx != first_pass[-1]:
                first_pass.append( prob_idx )
                first_pass_score.append( probs[idx][prob_idx] )
        
        second_pass = []
        second_pass_score = []
        for idx in range(len(first_pass)):
            first_pass_idx = first_pass[idx]
            first_pass_score_idx = first_pass_score[idx]
            if first_pass_idx != blank_id:
                second_pass.append( first_pass_idx )
                second_pass_score.append( first_pass_score_idx )
        
        return second_pass, second_pass_score

    tmp = preprocess(img)
    img = tmp.astype(np.float32)
    # print(img.shape)
    # img = img.transpose((2, 0, 1))
    
    net.blobs['data'].data[...] = img
    preds = net.forward()["reshape"]
    # print(preds.shape, len(preds[0][0][0]))

    preds = np.transpose(np.squeeze(preds))
    greedy_res, result_scors_list = greedy_decode(preds)
    result_ocr = ''.join([ocr_labels[greedy_res[idx]] for idx in range(len(greedy_res))])

    # result_ocr = ""
    # result_scors_list = []
    # out = preds[0].transpose((1, 2, 0))
    # for i in range(len(preds[0][0][0])):
    #     index=np.argmax(out[:][0][i])
    #     # print(out[:][0][i])
    #     # print(index)
    #     if index != 0:
    #         result_scors_list.append(out[:][0][i][index])
    #     result_ocr=result_ocr+ocr_labels[index]
    return result_ocr, result_scors_list


def license_palte_crnn_recognition_beamsearch_caffe(net, img, asr_beamsearch, lm):
    tmp = preprocess(img)
    img = tmp.astype(np.float32)
    
    net.blobs['data'].data[...] = img
    preds = net.forward()["reshape"]
    
    net_output = preds[0].transpose((1, 2, 0))

    net_output = torch.from_numpy(np.squeeze(net_output)).log()
    symbol_list = asr_beamsearch.prefix_beam_search(net_output, lm=lm)

    return ''.join(symbol_list)


if __name__ == '__main__':

    # china: license_plate_recognition_moel_lxn
    # caffe_prototxt = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china.prototxt"
    caffe_prototxt = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china_softmax.prototxt"
    caffe_model = "/mnt/huanyuan2/model/image_model/license_plate_recognition_moel_lxn/china.caffemodel"

    # image_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone_lincense_plate/Crop_itering/height_35_200/plate/"
    # image_list = os.listdir(image_dir)

    image_dir = "/mnt/huanyuan2/data/image/RM_ADAS_AllInOne/allinone_lincense_plate/Crop_itering/height_35_200/fuzzy_plate/"
    image_list = os.listdir(image_dir)

    net = license_palte_model_init_caffe(caffe_prototxt, caffe_model)

    for idx in range(len(image_list)):
        image_path = os.path.join(image_dir, image_list[idx])
        img = cv2.imread(image_path, 0) 
        result_ocr, result_scors_list = license_palte_crnn_recognition_caffe(net, img)
        print (image_list[idx], result_ocr, np.array(result_scors_list).mean())

