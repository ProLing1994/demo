from mmdet.apis import init_detector, inference_detector


class MmdetModel(object):
    def __init__(self, config_file=None, checkpoint_file=None, device="cpu"):
        if config_file is None:
            config_file = "ssd_detector/cascade_rcnn_x101_64x4d_fpn_1x.py"
        if checkpoint_file is None:
            checkpoint_file = "ssd_detector/cascade_rcnn_x101_64x4d_fpn_1x_epoch_4.pth"
        self.model = init_detector(config_file, checkpoint_file, device=device)

    def detect(self, img, thresh=0.2, with_score=False):
        result = inference_detector(self.model, img)
        # print(result)
        out_dict = {}
        for i, cls in enumerate(self.model.CLASSES):
            # print(cls)
            bbox_score = filter(lambda s: s[-1] > thresh, result[i].tolist())
            if with_score:
                bbox = list(map(lambda s: [int(j + 0.5) for j in s[:-1]] + [s[-1]], bbox_score))
            else:
                bbox = list(map(lambda s: [int(j + 0.5) for j in s[:-1]], bbox_score))
            out_dict[cls] = bbox
        return out_dict

# def mm_detector(frame):
#     config_file = "ssd_detector/cascade_rcnn_x101_64x4d_fpn_1x.py"
#     checkpoint_file = "ssd_detector/cascade_rcnn_x101_64x4d_fpn_1x_epoch_4.pth"
#     model = init_detector(config_file, checkpoint_file, device='cuda:0')
#     result = inference_detector(model, frame)
#     print(result)
