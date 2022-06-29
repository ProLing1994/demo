import cv2
import numpy as np
import sys 

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Demo.face_capture.face.model.decode import mot_decode, _tranpose_and_gather_feat
from Image.Demo.face_capture.face.model.vgg import *

class FairMot(object):
    def __init__(self, model_path, image_width, image_height):
        self.model_path = model_path

        # params
        self.image_width = image_width
        self.image_height = image_height
        self.resize_img_width = 320
        self.resize_img_height = 320
        
        self.num_classes = 1
        self.k = 128
        self.conf_thres = 0.3

        self.model_init()


    def model_init(self):
        # fairmot
        self.model = get_vggnet()
        self.checkpoint = torch.load(self.model_path, map_location=lambda storage, loc: storage)      
        self.model.load_state_dict(self.checkpoint['state_dict'], strict=False)
        self.model.eval()
        self.model = self.model.cuda()
    

    def detect(self, img_ori):
        # resize
        img = cv2.resize(img_ori, (self.resize_img_height, self.resize_img_width)).astype(np.float32)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # forward
        with torch.no_grad():
            im_blob = torch.from_numpy(img).cuda().unsqueeze(0)
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()            ## [1,1,80,80]
            wh = output['wh']                       ## [1,4,80,80]
            id_feature = output['id']               ## [1,128,80,80]
            id_feature = F.normalize(id_feature, dim=1)
            reg = output['reg']                     ## [1,2,80,80]
            
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=True, K=self.k)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()
        
        dets = self.post_process(dets)
        dets = self.merge_outputs([dets])[1]
        remain_inds = dets[:, 4] > self.conf_thres

        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]
        return dets, id_feature


    def post_process(self, dets):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])

        ret = []
        for i in range(dets.shape[0]):
            top_preds = {}
            # # TODO(huanyuan)，该模式联咏后处理不支持，需重新训练模型
            dets[i, :, 0] = dets[i, :, 0] * self.image_width / 80.0
            dets[i, :, 2] = dets[i, :, 2] * self.image_width / 80.0
            dets[i, :, 1] = dets[i, :, 1] * self.image_height / 80.0
            dets[i, :, 3] = dets[i, :, 3] * self.image_height / 80.0
            classes = dets[i, :, -1]
            for j in range(self.num_classes):
                inds = (classes == j)
                top_preds[j + 1] = np.concatenate([
                    dets[i, inds, :4].astype(np.float32),
                    dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
                ret.append(top_preds)
        
        for j in range(1, self.num_classes + 1):
            ret[0][j] = np.array(ret[0][j], dtype=np.float32).reshape(-1, 5)
        return ret[0]


    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.k:
            kth = len(scores) - self.k
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results
