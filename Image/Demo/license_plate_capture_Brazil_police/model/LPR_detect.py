import cv2
import numpy as np
import sys

# caffe_root = '/home/huanyuan/code/caffe_ssd-ssd/'
caffe_root = '/home/huanyuan/code/caffe_ssd-ssd-gpu/'
sys.path.insert(0, caffe_root + 'python')
import caffe


class LPRDetectCaffe(object):

    def __init__(self, prototxt, model_path, class_name=['license_plate'], input_shape=(300, 300), gpu_bool=False, conf_thres=0.4):

        self.prototxt = prototxt
        self.model_path = model_path
        self.input_shape = input_shape
        self.gpu_bool = gpu_bool

        self.conf_thres = conf_thres
        self.classes = ['__background__']
        self.classes.extend(class_name)
        self.model_init()


    def model_init(self):

        if self.gpu_bool:
            caffe.set_device(0)
            caffe.set_mode_gpu()
            print("[Information:] GPU mode")
        else:
            caffe.set_mode_cpu()
            print("[Information:] CPU mode")

        self.net = caffe.Net(self.prototxt, self.model_path, caffe.TEST)


    def model_forward(self, img):
        self.net.blobs['data'].data[...] = img
        out = self.net.forward()

        h = self.image_height
        w = self.image_width
        box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])
        cls = out['detection_out'][0, 0, :, 1]
        conf = out['detection_out'][0, 0, :, 2]
        return (box.astype(np.int32), conf, cls)


    def preprocess(self, src):
        img = cv2.resize(src, self.input_shape)
        img = img.astype(np.float32)
        rgb_mean = np.array((104, 117, 123), dtype=np.int)
        img -= rgb_mean
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        # img = img / 255.0
        return img

    
    def postprocess(self, box, conf, cls, with_score):

        out_dict = {}

        for idx in range(len(box)):
            if conf[idx] > self.conf_thres:
                box_name = self.classes[int(cls[idx])]
                box_scores = conf[idx]
                box_locations = (int(box[idx][0]), int(box[idx][1]), int(box[idx][2]), int(box[idx][3])) 
                l, t, r, b = box_locations 

                if box_name not in out_dict:
                    out_dict[box_name] = []

                if with_score:
                    out_dict[box_name].append([l, t, r, b, box_scores])
                else:
                    out_dict[box_name].append([l, t, r, b])
                
        return out_dict



    def detect(self, img, with_score=False):

        # info 
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        
        # preprocess
        img = self.preprocess(img)

        # forward
        box, conf, cls = self.model_forward(img)

        # postprocess
        detect_res_dict = self.postprocess(box, conf, cls, with_score)

        return detect_res_dict


class LPRDetectOpenVINO(object):
    
    def __init__(self, model_path, input_shape=(300, 300), gpu_bool=False):

        self.model_path = model_path
        self.input_shape = input_shape
        self.gpu_bool = gpu_bool

        self.conf_thres = 0.4
        self.classes = ('__background__', 'license_plate') 
        self.model_init()


    def model_init(self):
        
        # api: Inference Engine API
        import openvino.inference_engine as ie
        # Inference Engine API
        core = ie.IECore()

        # Read a model from a drive
        network = core.read_network(self.model_path)

        # Load the Model to the Device
        self.exec_network = core.load_network(network, "CPU")
        self.input_blob = next(iter(network.input_info))

        # # api: OpenVINO™ Runtime API 2.0:
        # from openvino.runtime import Core
        # # Load the Model
        # ie = Core()
        # model = ie.read_model(model=self.model_path)

        # if self.gpu_bool:
        #     self.compiled_model = ie.compile_model(model=model, device_name="GPU")
        # else:
        #     self.compiled_model = ie.compile_model(model=model, device_name="CPU")
        # self.input_layer_ir = next(iter(self.compiled_model.inputs))

        return 


    def model_forward(self, img):

        h = self.image_height
        w = self.image_width
        img = np.expand_dims(img, 0)

        # Start Inference
        # api: Inference Engine API
        out = self.exec_network.infer(inputs={self.input_blob: img})
        box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])
        cls = out['detection_out'][0, 0, :, 1]
        conf = out['detection_out'][0, 0, :, 2]

        # # api: OpenVINO™ Runtime API 2.0:
        # request = self.compiled_model.create_infer_request()
        # request.infer({self.input_layer_ir.any_name: img})
        # box = request.get_tensor("detection_out").data[0, 0, :, 3:7] * np.array([w, h, w, h])
        # cls = request.get_tensor("detection_out").data[0, 0, :, 1]
        # conf = request.get_tensor("detection_out").data[0, 0, :, 2]
        
        return (box.astype(np.int32), conf, cls)


    def preprocess(self, src):
        img = cv2.resize(src, self.input_shape)
        img = img.astype(np.float32)
        rgb_mean = np.array((104, 117, 123), dtype=np.int)
        img -= rgb_mean
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        # img = img / 255.0
        return img

    
    def postprocess(self, box, conf, cls, with_score):

        out_dict = {}

        for idx in range(len(box)):
            if conf[idx] > self.conf_thres:
                box_name = self.classes[int(cls[idx])]
                box_scores = conf[idx]
                box_locations = (int(box[idx][0]), int(box[idx][1]), int(box[idx][2]), int(box[idx][3])) 
                l, t, r, b = box_locations 

                if box_name not in out_dict:
                    out_dict[box_name] = []

                if with_score:
                    out_dict[box_name].append([l, t, r, b, box_scores])
                else:
                    out_dict[box_name].append([l, t, r, b])
                
        return out_dict



    def detect(self, img, with_score=False):

        # info 
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        
        # preprocess
        img = self.preprocess(img)

        # forward
        box, conf, cls = self.model_forward(img)

        # postprocess
        detect_res_dict = self.postprocess(box, conf, cls, with_score)

        return detect_res_dict


if __name__ == '__main__':

    # caffe
    # ssd_plate_prototxt = "/mnt/huanyuan/model_final/image_model/zd_ssd_rfb_wmr/ssd_mbv2_2class/caffe_model/ssd_mobilenetv2_fpn.prototxt"
    # ssd_plate_model_path = "/mnt/huanyuan/model_final/image_model/zd_ssd_rfb_wmr/ssd_mbv2_2class/caffe_model/ssd_mobilenetv2_0421.caffemodel"
    ssd_plate_model_path = "/mnt/huanyuan/model_final/image_model/zd_ssd_rfb_wmr/ssd_mbv2_2class/openvino_model/ssd_mobilenetv2_fpn.xml"

    img_path = "/mnt/huanyuan2/data/image/ZD_anpr/test_video/ZD_DUBAI/avi文件/5M_白天_侧向_0615/jpg/0000000000000003-220615-061710-061722-00050E226151/0000000000000003-220615-061710-061722-00050E226151_00200.jpg"
    output_path = "/mnt/huanyuan2/data/image/ZD_anpr/test_video/ZD_DUBAI/avi文件/5M_白天_侧向_0615/jpg/test.jpg"

    # init
    # detector = LPRDetectCaffe(ssd_plate_prototxt, ssd_plate_model_path)
    detector = LPRDetectOpenVINO(ssd_plate_model_path)

    # img
    img = cv2.imread(img_path)

    bboxes = detector.detect( img, with_score=True )

    for idy in range(len(bboxes['license_plate'])): 
        bboxes_idx = bboxes['license_plate'][idy]

        p1 = (int(bboxes_idx[0]), int(bboxes_idx[1]))
        p2 = (int(bboxes_idx[2]), int(bboxes_idx[3]))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)

        p3 = (int((bboxes_idx[0] + bboxes_idx[2])/2), int((bboxes_idx[1] + bboxes_idx[3])/2))
        title = "%s:%.2f" % ('license_plate', bboxes_idx[-1])
        cv2.putText(img, title, p3, cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

    cv2.imwrite(output_path, img)



