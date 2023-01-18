import cv2
import numpy as np
import math
import sys
import yaml

# import paddle
# sys.path.insert(0, '/yuanhuan/code/demo/Image/recognition2d/PaddleOCR')
# from ppocr.data import create_operators, transform
# from ppocr.modeling.architectures import build_model
# from ppocr.postprocess import build_post_process
# from ppocr.utils.save_load import load_model
# from tools.program import load_config, merge_config

caffe_root = '/home/huanyuan/code/caffe_ssd-ssd/'
# caffe_root = '/home/huanyuan/code/caffe_ssd-ssd-gpu/'
sys.path.insert(0, caffe_root + 'python')
import caffe


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


class LPRPaddle(object):
    
    def __init__(self, config_path, model_path):

        self.config_path = config_path
        self.model_path = model_path
        self.gpu_bool = True

        self.config_init()
        self.device_init()
        self.post_process_init()
        self.model_init()
        self.data_opts_init()
        self.ocr_labels_init()
    

    def config_init(self):

        self.config = load_config(self.config_path)
        self.config = merge_config(self.config, {"Global.pretrained_model": yaml.load(self.model_path, Loader=yaml.Loader)})
        self.global_config = self.config['Global']

    def device_init(self):

        self.device = 'cpu'
        if self.gpu_bool:
             self.device = 'gpu:0'
        self.device = paddle.set_device(self.device)


    def post_process_init(self):

        # build post process
        self.post_process_class = build_post_process(self.config['PostProcess'],
                                                    self.global_config)

    def model_init(self):
        
        # build model
        if hasattr(self.post_process_class, 'character'):
            char_num = len(getattr(self.post_process_class, 'character'))
            if self.config['Architecture']["algorithm"] in ["Distillation",
                                                    ]:  # distillation model
                for key in self.config['Architecture']["Models"]:
                    if self.config['Architecture']['Models'][key]['Head'][
                            'name'] == 'MultiHead':  # for multi head
                        out_channels_list = {}
                        if self.config['PostProcess'][
                                'name'] == 'DistillationSARLabelDecode':
                            char_num = char_num - 2
                        out_channels_list['CTCLabelDecode'] = char_num
                        out_channels_list['SARLabelDecode'] = char_num + 2
                        self.config['Architecture']['Models'][key]['Head'][
                            'out_channels_list'] = out_channels_list
                    else:
                        self.config['Architecture']["Models"][key]["Head"][
                            'out_channels'] = char_num
            elif self.config['Architecture']['Head'][
                    'name'] == 'MultiHead':  # for multi head loss
                out_channels_list = {}
                if self.config['PostProcess']['name'] == 'SARLabelDecode':
                    char_num = char_num - 2
                out_channels_list['CTCLabelDecode'] = char_num
                out_channels_list['SARLabelDecode'] = char_num + 2
                self.config['Architecture']['Head'][
                    'out_channels_list'] = out_channels_list
            else:  # base rec model
                self.config['Architecture']["Head"]['out_channels'] = char_num

        self.model = build_model(self.config['Architecture'])

        # 计算参数量和 FLOPS
        # paddle.summary(self.model, (1, 3, 32, 320))
        # FLOPs = paddle.flops(self.model, input_size=[1, 3, 32, 320], print_detail=True)
        # paddle.summary(self.model, (1, 3, 64, 256))
        # FLOPs = paddle.flops(self.model, input_size=[1, 3, 64, 256], print_detail=True)
        # paddle.summary(self.model, (1, 1, 64, 320))
        # FLOPs = paddle.flops(self.model, input_size=[1, 1, 64, 320], print_detail=True)

        load_model(self.config, self.model)

        self.model.eval()


    def data_opts_init(self):   

        # create data ops
        transforms = []
        for op in self.config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name in ['RecResizeImg']:
                op[op_name]['infer_mode'] = True
            elif op_name == 'KeepKeys':
                if self.config['Architecture']['algorithm'] == "SRN":
                    op[op_name]['keep_keys'] = [
                        'image', 'encoder_word_pos', 'gsrm_word_pos',
                        'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                    ]
                elif self.config['Architecture']['algorithm'] == "SAR":
                    op[op_name]['keep_keys'] = ['image', 'valid_ratio']
                elif self.config['Architecture']['algorithm'] == "RobustScanner":
                    op[op_name][
                        'keep_keys'] = ['image', 'valid_ratio', 'word_positons']
                else:
                    op[op_name]['keep_keys'] = ['image']
            transforms.append(op)
        self.global_config['infer_mode'] = True
        self.ops = create_operators(transforms, self.global_config)


    def ocr_labels_init(self):
        
        self.character_str = []
        with open(self.global_config['character_dict_path'], "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        if self.global_config['use_space_char']:
            self.character_str.append(" ")
        self.ocr_labels = list(self.character_str)


    def run(self, img):

        data = {'image': img}
        batch = transform(data, self.ops)
        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)
        preds = self.model(images)

        # post_result = self.post_process_class(preds)
        # result_ocr, result_scors = post_result[0][0], post_result[0][1]

        if isinstance(preds, dict):
            result_str, result_scors = greedy_decode(preds['Student']['head_out'].numpy()[0])
        else:
            result_str, result_scors = greedy_decode(preds.numpy()[0])
        result_ocr = ''.join([self.ocr_labels[result_str[idx] - 1] for idx in range(len(result_str))])

        return result_ocr, result_scors


class LPROnnx(object):
    
    def __init__(self, config_path, model_path):

        self.config_path = config_path
        self.model_path = model_path
        self.gpu_bool = True
        self.bool_white = True
        # self.img_shape = (3, 64, 256)
        # self.img_shape = (1, 64, 256)
        self.img_shape = (1, 64, 320)

        self.config_init()
        self.model_init()
        self.ocr_labels_init()
    

    def config_init(self):

        self.config = load_config(self.config_path)
        self.config = merge_config(self.config, {"Global.pretrained_model": yaml.load(self.model_path, Loader=yaml.Loader)})
        self.global_config = self.config['Global']


    def model_init(self):
        
        # build model
        import onnxruntime as ort
        self.model = ort.InferenceSession(self.model_path)


    def ocr_labels_init(self):
        
        self.character_str = []
        with open(self.global_config['character_dict_path'], "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        if self.global_config['use_space_char']:
            self.character_str.append(" ")
        self.ocr_labels = list(self.character_str)


    def preprocess(self, img, image_shape):
        imgC, imgH, imgW = image_shape
        # todo: change to 0 and modified image shape
        max_wh_ratio = imgW * 1.0 / imgH
        h, w = img.shape[0], img.shape[1]
        ratio = w * 1.0 / h
        max_wh_ratio = min(max(max_wh_ratio, ratio), max_wh_ratio)
        imgW = int(imgH * max_wh_ratio)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im


    def preprocess_rm(self, img, image_shape):

        imgC, imgH, imgW = image_shape

        if imgC == 1:
            resized_image = cv2.resize(img, (imgW, imgH))

            # gray
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            resized_image = resized_image[:, :, np.newaxis]
        else:
            resized_image = cv2.resize(img, (imgW, imgH))

        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        return resized_image


    def preprocess_rm_ratio(self, img, image_shape):
        
        img = img[: img.shape[0]//4*4, : img.shape[1]//4*4, :]

        h, w = img.shape[0], img.shape[1]
        imgC, imgH, imgW = image_shape

        max_wh_ratio = imgW * 1.0 / imgH
        ratio = w * 1.0 / h
      
        # pad
        if ratio < max_wh_ratio:
            to_imgW = int(math.ceil(h * max_wh_ratio)) // 4 * 4
            if (to_imgW - w < 32):
                to_imgW = w + 32

            if self.bool_white:
                pad_img = np.ones((h, to_imgW, 3), dtype=np.uint8)
                pad_img *= 255
            else:
                pad_img = np.zeros((h, to_imgW, 3), dtype=np.uint8)

            pad_img[:, 0:w, :] = img  
            img = pad_img  
        else:
            img = img

        if imgC == 1:
            resized_image = cv2.resize(img, (imgW, imgH))

            # gray
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            resized_image = resized_image[:, :, np.newaxis]
        else:
            resized_image = cv2.resize(img, (imgW, imgH))

        resized_image.astype(np.float32)
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0

        return resized_image


    def run(self, img):

        # img = self.preprocess(img, self.img_shape)
        # img = self.preprocess_rm(img, self.img_shape)
        img = self.preprocess_rm_ratio(img, self.img_shape)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        input_dict = {}
        input_dict[self.model.get_inputs()[0].name] = img
        preds = self.model.run(None, input_dict)[0]

        result_str, result_scors = greedy_decode(preds[0])
        result_ocr = ''.join([self.ocr_labels[result_str[idx] - 1] for idx in range(len(result_str))])

        return result_ocr, result_scors


class LPRCaffe(object):
    
    def __init__(self, model_path, prototxt_path, dict_path, gpu_bool=False):

        self.model_path = model_path
        self.prototxt_path = prototxt_path
        self.dict_path = dict_path
        self.gpu_bool = gpu_bool
        self.bool_white = True
        # self.img_shape = (3, 64, 256)
        # self.img_shape = (1, 64, 256)
        self.img_shape = (1, 64, 320)

        self.model_init()
        self.ocr_labels_init()


    def model_init(self):
        
        # build model
        if self.gpu_bool:
            caffe.set_device(0)
            caffe.set_mode_gpu()
            print("[Information:] GPU mode")
        else:
            caffe.set_mode_cpu()
            print("[Information:] CPU mode")

        self.net = caffe.Net(self.prototxt_path, self.model_path, caffe.TEST)


    def ocr_labels_init(self):
        
        self.character_str = []
        with open(self.dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
            self.character_str.append(" ")
        self.ocr_labels = list(self.character_str)


    def preprocess_rm(self, img, image_shape):

        imgC, imgH, imgW = image_shape

        if imgC == 1:
            resized_image = cv2.resize(img, (imgW, imgH))

            # gray
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            resized_image = resized_image[:, :, np.newaxis]
        else:
            resized_image = cv2.resize(img, (imgW, imgH))

        resized_image = resized_image.transpose((2, 0, 1)) / 255.0
        return resized_image


    def preprocess_rm_ratio(self, img, image_shape):
        
        img = img[: img.shape[0]//4*4, : img.shape[1]//4*4, :]

        h, w = img.shape[0], img.shape[1]
        imgC, imgH, imgW = image_shape

        max_wh_ratio = imgW * 1.0 / imgH
        ratio = w * 1.0 / h
      
        # pad
        if ratio < max_wh_ratio:
            to_imgW = int(math.ceil(h * max_wh_ratio)) // 4 * 4
            if (to_imgW - w < 32):
                to_imgW = w + 32

            if self.bool_white:
                pad_img = np.ones((h, to_imgW, 3), dtype=np.uint8)
                pad_img *= 255
            else:
                pad_img = np.zeros((h, to_imgW, 3), dtype=np.uint8)

            pad_img[:, 0:w, :] = img  
            img = pad_img  
        else:
            img = img

        if imgC == 1:
            resized_image = cv2.resize(img, (imgW, imgH))

            # gray
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            resized_image = resized_image[:, :, np.newaxis]
        else:
            resized_image = cv2.resize(img, (imgW, imgH))

        resized_image.astype(np.float32)
        resized_image = resized_image.transpose((2, 0, 1)) / 255.0

        return resized_image


    def run(self, img):

        # img = self.preprocess_rm(img, self.img_shape)
        img = self.preprocess_rm_ratio(img, self.img_shape)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        self.net.blobs['x'].data[...] = img
        preds = self.net.forward()["probs"]
        preds = np.transpose(np.squeeze(preds))

        result_str, result_scors = greedy_decode(preds)
        result_ocr = ''.join([self.ocr_labels[result_str[idx] - 1] for idx in range(len(result_str))])

        return result_ocr, result_scors


if __name__ == '__main__': 

    ###############################
    # paddle
    ###############################

    # config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_simple/config.yml"
    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_simple/best_accuracy"

    # config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_simple/config.yml"
    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_simple/best_accuracy"

    # config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray_64_320_1215_simple/config.yml"
    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray_64_320_1215_simple/best_accuracy"

    # config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all/config.yml"
    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all/best_accuracy"

    # config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_all/config.yml"
    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_all/best_accuracy"

    # config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray_64_320_1215_all/config.yml"
    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_center_rmresize_ratio_gray_64_320_1215_all/best_accuracy"

    # config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all_aug/config.yml"
    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_gray_64_320_1215_all_aug/best_accuracy"

    # config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_all_aug/config.yml"
    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_gray_64_320_1215_all_aug/best_accuracy"

    # config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_1219_all_aug/config.yml"
    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_1219_all_aug/best_accuracy"

    # config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_white_gray_64_320_1220_all_aug/config.yml"
    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_gtc_rmresize_ratio_white_gray_64_320_1220_all_aug/best_accuracy"

    config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_end_64_384_1226_all_aug/config.yml"
    model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_end_64_384_1226_all_aug/best_accuracy"

    # img_path = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_paddle_ocr/train_data/rec/test/0000000000000000-220804-131737-131833-00000D000150_26.56_43.63-sn00070-00_none_none_none_Double_J#18886.jpg"
    img_path = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_paddle_ocr/test/crop.jpg"

    lpr_paddle = LPRPaddle(config_path, model_path)

    with open(img_path, 'rb') as f:
        img = f.read()
    res_ocr, res_scors = lpr_paddle.run(img)
    print(res_ocr, res_scors)

    # ###############################
    # # onnx
    # ###############################
    # config_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_1219_all_aug/config.yml"
    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_1219_all_aug/inference/onnx/model.onnx"

    # # img_path = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_paddle_ocr/train_data/rec/test/0000000000000000-220804-131737-131833-00000D000150_26.56_43.63-sn00070-00_none_none_none_Double_J#18886.jpg"
    # img_path = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_paddle_ocr/test/crop.jpg"

    # lpr_onnx = LPROnnx(config_path, model_path)
    
    # img = cv2.imread(img_path)
    # res_ocr, res_scors = lpr_onnx.run(img)
    # print(res_ocr, res_scors)
    
    # ###############################
    # # caffe
    # ###############################

    # model_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_1219_all_aug/inference/caffe/model-sim.clip.rename.caffemodel"
    # prototxt_path = "/yuanhuan/model/image/lpr/paddle_ocr/v1_en_number_mobilenet_v1_rm_cnn_tc_res_mobile_rmresize_ratio_white_gray_64_320_1219_all_aug/inference/caffe/model-sim.clip.rename.prototxt"
    # dict_path = "/yuanhuan/data/image/LicensePlate_ocr/original/zd/UAE/type/zd_dict.txt"

    # # img_path = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_paddle_ocr/train_data/rec/test/0000000000000000-220804-131737-131833-00000D000150_26.56_43.63-sn00070-00_none_none_none_Double_J#18886.jpg"
    # img_path = "/yuanhuan/data/image/LicensePlate_ocr/training/plate_zd_mask_paddle_ocr/test/crop.jpg"

    # lpr_caffe = LPRCaffe(model_path, prototxt_path, dict_path)
    
    # img = cv2.imread(img_path)
    # res_ocr, res_scors = lpr_caffe.run(img)
    # print(res_ocr, res_scors)