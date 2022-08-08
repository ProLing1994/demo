import cv2
from collections import OrderedDict
import sys

from api.ssd.config import *
from api.ssd.nms import nms


class SSDDetector(object):

    def __init__(self, threshold=0.4, prototxt=None, model_path="ssd_detector/SSD_VGG_FPN_VOC_epoches_165.pth", ssd_caffe_bool=False, ssd_openvino_bool=False, merge_class_bool=False, gpu_bool=False):
        self.cfg = VOC_300
        self.img_dim = 300
        self.fpn_type = 'FPN'
        self.gpu_bool = gpu_bool

        self.prototxt = prototxt
        self.model_path = model_path
        self.num_classes = num_classes
        self.ssd_caffe_bool = ssd_caffe_bool
        self.ssd_openvino_bool = ssd_openvino_bool
        self.attri_num_classes = attri_num_classes
        self.merge_class_bool = merge_class_bool

        self.threshold = threshold
        self.top_k = 200

        self.net = self.load_net()

    def load_net(self):
        if self.ssd_caffe_bool:
            caffe_root = '/home/huanyuan/code/caffe/'
            sys.path.insert(0, caffe_root+'python')
            import caffe

            from prior_box_cpu import PriorBox
            priorbox = PriorBox(self.cfg)
            self.priors = priorbox.forward()

            return self.caffe_load_net()
        elif self.ssd_openvino_bool:
            from api.ssd.prior_box_cpu import PriorBox
            priorbox = PriorBox(self.cfg)
            self.priors = priorbox.forward()

            return self.openvino_load_net()
        else:
            import torch             
            import data.data_augment 
            from layers.functions.detection import Detect
            from layers.functions.prior_box import PriorBox
            
            priorbox = PriorBox(self.cfg)
            with torch.no_grad():
                self.priors = priorbox.forward()
                if self.gpu_bool:
                    self.priors = self.priors.cuda()

            self.detector = Detect(num_classes, attri_num_classes, 0, self.cfg)
            return self.pytorch_load_net()
    
    def caffe_load_net(self):
        if self.gpu_bool:
            # caffe.set_device(0)
            # caffe.set_mode_gpu()
            print("[Information:] GPU mode")
        else:
            caffe.set_mode_cpu()
            print("[Information:] CPU mode")

        net = caffe.Net(self.prototxt, self.model_path, caffe.TEST)
        return net


    def openvino_load_net(self):
        # # api: Inference Engine API
        # import openvino.inference_engine as ie
        # # Inference Engine API
        # core = ie.IECore()

        # # Read a model from a drive
        # network = core.read_network(self.model_path)

        # # Load the Model to the Device
        # exec_network = core.load_network(network, "CPU")
        # input_blob = next(iter(network.input_info))

        # return (exec_network, input_blob)

        # api: OpenVINO™ Runtime API 2.0:
        from openvino.runtime import Core
        # Load the Model
        ie = Core()
        model = ie.read_model(model=self.model_path)

        if self.gpu_bool:
            compiled_model = ie.compile_model(model=model, device_name="GPU")
        else:
            compiled_model = ie.compile_model(model=model, device_name="CPU")
        input_layer_ir = next(iter(compiled_model.inputs))

        return (compiled_model, input_layer_ir)


    def pytorch_load_net(self):
        from models.SSD_VGG_Optim_FPN_RFB import build_net
        import torch
        import torch.backends.cudnn as cudnn

        net = build_net('test', self.img_dim, self.num_classes, self.attri_num_classes, self.fpn_type)  # initialize detector
        state_dict = torch.load(self.model_path)

        # load state
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.eval()
        print('Finished loading model!')
        print(net)

        if self.gpu_bool:
            net = net.cuda()
            cudnn.benchmark = True
        else:
            net = net.cpu()
        return net

    def detect(self, img, with_score=False):
        if self.ssd_caffe_bool:
            all_boxes = self._detect_caffe(img, 
                                    self.net,
                                    thresh=self.threshold)
        elif self.ssd_openvino_bool:
            all_boxes = self._detect_openvino(img, 
                                    self.net,
                                    thresh=self.threshold)
        else:
            from data.data_augment import BaseTransform
            all_boxes = self._detect_pytorch(img, 
                                    self.net, 
                                    self.detector, 
                                    self.gpu_bool,
                                    BaseTransform(self.img_dim, rgb_means, (2, 0, 1)), 
                                    self.top_k, 
                                    thresh=self.threshold)
        
        h, w, _ = img.shape
        out_dict = {}
        for k in range(1, num_classes):
            box_locations = []
            if len(all_boxes[k]) > 0:
                box_scores = all_boxes[k][:, 4]
                box_locations = all_boxes[k][:, 0:4]
                box_locations = [[int(b + 0.5) for b in box] for box in box_locations]

                if pos_CLASSES[k] == attri_label:
                    box_attris_scores = all_boxes[k][:, 6]
                    box_attris = all_boxes[k][:, 5]

                    if self.merge_class_bool:
                        box_scores = box_scores * box_attris_scores

                bbox_out = []
                for box, score in zip(box_locations, box_scores):
                    l, t, r, b = box
                    l = max(l, 0)
                    t = max(t, 0)
                    r = min(w - 1, r)
                    b = min(h - 1, b)
                    if with_score:
                        bbox_out.append([l, t, r, b, score])
                    else:
                        bbox_out.append([l, t, r, b])
                box_locations = bbox_out

                if self.merge_class_bool:
                    out_dict[pos_CLASSES[k]] = box_locations
                else:
                    if pos_CLASSES[k] == attri_label:
                        for attris_idx in range(len(box_attris)):
                            box_attris_idx = box_attris[attris_idx]
                            out_dict[attri_CLASSSES[int(box_attris_idx)]] = list(np.array(box_locations)[box_attris == box_attris_idx])
                    else:
                        out_dict[pos_CLASSES[k]] = box_locations
        
        return out_dict

    def _detect_caffe(self, img_ori, net, thresh=0.5):

        def preprocess(src):
            img = cv2.resize(src, (self.img_dim, self.img_dim)).astype(np.float32)
            rgb_mean = np.array(rgb_means, dtype=np.int)
            img -= rgb_mean
            img = img.astype(np.float32)
            return img

        def decode(loc, priors, variances):
            """Decode locations from predictions using priors to undo
            the encoding we did for offset regression at train time.
            Args:
                loc: location predictions for loc layers,
                    Shape: [num_priors,4]
                priors: Prior boxes in center-offset form.
                    Shape: [num_priors,4].
                variances: (list[float]) Variances of priorboxes
            Return:
                decoded bounding box predictions
            """

            boxes = np.concatenate((
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]
            #print(boxes)
            return boxes

        def postprocess(img, out):
            h = img.shape[0]
            w = img.shape[1]
            boxes = out['mbox_loc_reshape'][0]
            boxes = decode(boxes, self.priors, self.cfg['variance'])
            boxes *= np.array([w, h, w, h])
            scores = out['mbox_conf_reshape'][0]
            attris = out['mbox_attri_reshape'][0]
            return (boxes.astype(np.int32), scores, attris)

        img = preprocess(img_ori)

        img = img.transpose((2, 0, 1))
        net.blobs['data'].data[...] = img
        out = net.forward()

        boxes, scores, attris = postprocess(img_ori, out)

        # dets
        all_boxes = [[] for _ in range(num_classes)]

        for j in range(1, num_classes):
    
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.empty([0, 5], dtype=np.float32)
                continue

            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_attris = attris[inds]

            if pos_CLASSES[j] == attri_label:
                attri_inds = np.argmax(c_attris, axis = 1)
                attri_scores = np.max(c_attris, axis = 1)
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis], attri_inds[:, np.newaxis], attri_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)
            else:
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)

            keep = nms(c_dets, 0.45)
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets

        return all_boxes

    def _detect_openvino(self, img_ori, net, thresh=0.5):

        def preprocess(src):
            img = cv2.resize(src, (self.img_dim, self.img_dim)).astype(np.float32)
            rgb_mean = np.array(rgb_means, dtype=np.int)
            img -= rgb_mean
            img = img.astype(np.float32)
            return img

        def decode(loc, priors, variances):
            """Decode locations from predictions using priors to undo
            the encoding we did for offset regression at train time.
            Args:
                loc: location predictions for loc layers,
                    Shape: [num_priors,4]
                priors: Prior boxes in center-offset form.
                    Shape: [num_priors,4].
                variances: (list[float]) Variances of priorboxes
            Return:
                decoded bounding box predictions
            """

            boxes = np.concatenate((
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]
            #print(boxes)
            return boxes

        def postprocess(img, boxes, scores, attris):
            h = img.shape[0]
            w = img.shape[1]
            boxes = decode(boxes, self.priors, self.cfg['variance'])
            boxes *= np.array([w, h, w, h])
            return (boxes.astype(np.int32), scores, attris)
        
        # api: Inference Engine API
        # exec_network, input_blob = net
        # api: OpenVINO™ Runtime API 2.0:
        compiled_model, input_layer_ir = net
        img = preprocess(img_ori)

        img = np.expand_dims(img.transpose(2, 0, 1), 0)

        # # Start Inference
        # # api: Inference Engine API
        # res = exec_network.infer(inputs={input_blob: img})
        # boxes = res['mbox_loc_reshape'][0]
        # scores = res['mbox_conf_reshape'][0]
        # attris = res['mbox_attri_reshape'][0]

        # api: OpenVINO™ Runtime API 2.0:
        request = compiled_model.create_infer_request()
        request.infer({input_layer_ir.any_name: img})
        boxes = request.get_tensor("mbox_loc_reshape").data[0]
        scores = request.get_tensor("mbox_conf_reshape").data[0]
        attris = request.get_tensor("mbox_attri_reshape").data[0]

        boxes, scores, attris = postprocess(img_ori, boxes, scores, attris)
    
        # dets
        all_boxes = [[] for _ in range(num_classes)]

        for j in range(1, num_classes):
    
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.empty([0, 5], dtype=np.float32)
                continue

            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_attris = attris[inds]

            if pos_CLASSES[j] == attri_label:
                attri_inds = np.argmax(c_attris, axis = 1)
                attri_scores = np.max(c_attris, axis = 1)
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis], attri_inds[:, np.newaxis], attri_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)
            else:
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)

            keep = nms(c_dets, 0.45)
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets

        return all_boxes

    def _detect_pytorch(self, img, net, detector, gpu_bool, transform, max_per_image=300, thresh=0.5):
        import torch
        
        scale = torch.Tensor([img.shape[1], img.shape[0],
                                img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if gpu_bool:
                x = x.cuda()
                scale = scale.cuda()

        out = net(x)      # forward pass
        boxes, scores, attris = detector.forward(out, self.priors)

        boxes = boxes[0]
        scores = scores[0]
        attris = attris[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        attris = attris.cpu().numpy()
        # scale each detection back up to the image
        
        # dets
        all_boxes = [[] for _ in range(num_classes)]

        for j in range(1, num_classes):

            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.empty([0, 5], dtype=np.float32)
                continue

            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_attris = attris[inds]

            if pos_CLASSES[j] == attri_label:
                attri_inds = np.argmax(c_attris, axis = 1)
                attri_scores = np.max(c_attris, axis = 1)
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis], attri_inds[:, np.newaxis], attri_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)
            else:
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                    np.float32, copy=False)

            keep = nms(c_dets, 0.45)
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets
        
        return all_boxes