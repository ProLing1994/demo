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


def preprocess(src):
    img = cv2.resize(src, (256, 64))
    img = img / 255.0
    return img


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


class LPR(object):

    def __init__(self, prototxt, model_path, prefix_beam_search_bool=False):

        self.prototxt = prototxt
        self.model_path = model_path
        self.prefix_beam_search_bool = prefix_beam_search_bool

        self.model_init()
    
    
    def model_init(self):
        
        self.net = caffe.Net(self.prototxt, self.model_path, caffe.TEST)

        if self.prefix_beam_search_bool:

            self.asr_beamsearch = Decode_BeamSearch.BeamSearch(ocr_labels, beam_size=5)
            self.lm = None
    

    def run(self, img):

        img = preprocess(img)
        img = img.astype(np.float32)

        self.net.blobs['data'].data[...] = img
        preds = self.net.forward()["probs"]
        preds = np.transpose(np.squeeze(preds))

        if self.prefix_beam_search_bool:
            
            result_symbol_list = self.asr_beamsearch.prefix_beam_search(torch.from_numpy(preds).log(), lm=self.lm)
            result_ocr = ''.join(result_symbol_list)

            return result_ocr
        else:

            # greedy 
            result_str, result_scors = greedy_decode(preds)
            result_ocr = ''.join([ocr_labels[result_str[idx]] for idx in range(len(result_str))])

            return result_ocr, result_scors



if __name__ == '__main__':

    # # china: lpr_lxn
    # caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr_lxn/china_softmax.prototxt"
    # caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr_lxn/china.caffemodel"
    
    # china: lpr_zg
    caffe_prototxt = "/mnt/huanyuan/model_final/image_model/lpr_zg/china/double/0628/china_double_softmax.prototxt"
    caffe_model_path = "/mnt/huanyuan/model_final/image_model/lpr_zg/china/double/0628/china_double_0628.caffemodel"

    # prefix_beam_search_bool = False
    prefix_beam_search_bool = True

    image_dir = "/mnt/huanyuan2/data/image/LicensePlateRecognition/test/"

    # init 
    lpr = LPR(caffe_prototxt, caffe_model_path, prefix_beam_search_bool)

    image_list = os.listdir(image_dir)
    image_list.sort()
    for idx in range(len(image_list)):
        image_name = image_list[idx]
        image_path = os.path.join(image_dir, image_name)

        if not image_name.endswith(".jpg"):
            continue

        img = cv2.imread(image_path, 0) 
        ocr = lpr.run(img)
        print( "{} -> {}".format(image_name, ocr))


