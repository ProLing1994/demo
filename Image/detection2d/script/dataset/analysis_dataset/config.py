# config.py
import numpy as np
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")
ddir = os.path.join(home,"data/VOCdevkit/")

# note: if you used our download scripts, this should be right

# VOCroot = "/mnt/huanyuan2/data/image/" # path to VOCdevkit root dir
VOCroot = "/yuanhuan/data/image/" # path to VOCdevkit root dir
# VOCroot = "/yuanhuan/data/image/minidata/" # path to VOCdevkit root dir

COCOroot = os.path.join(home, "data/COCO/")

VOC_CLASSES = ( '__background__', 'background_all', "motorcyclist", "license_plate", "moto_license_plate", "car", "bus", "truck", 'background_car_bus_truck', 'background_motorcyclist', 'background_license_plate', 'car_bus_truck', "car_o", "bus_o", "truck_o", "car_bus_truck_o", "motorcyclist_o", "license_plate_ignore", "moto_license_plate_ignore", 'neg')
ignore_CLASSES = ( "car_o", "bus_o", "truck_o", "car_bus_truck_o", "motorcyclist_o", "license_plate_ignore", "moto_license_plate_ignore" )
negtive_CLASSES = ('neg',)

# 1：background_all  8： background_car_bus_truck  9： background_motorcyclist  10： background_license_plate
negative_label = [1, 8, 9, 10]

# 最终网络输出的类别顺序：
pos_CLASSES = ('background_all', "motorcyclist", 'total_license_plate', 'car_bus_truck')
attri_CLASSSES = ('car', 'bus', 'truck')
attri_label = 'car_bus_truck'
attri_CLASSSES_license_plate = ('license_plate', 'moto_license_plate')
attri_label_license_plate = 'total_license_plate'
with_obl = False
# with_obl = True

# focal loss weight
attri_weight = np.array([[1/9, 0, 0], [0, 4/9, 0], [0, 0, 4/9]])

# eql class weight
# 'car', 'bus', 'truck'
eql_weight = np.array([1, 0, 0])

avoid_classes = [{8,9,10}, {8,9}, {8,10}, {8}, {9}, {10}]

rgb_means = (104, 117, 123)             # 实际这里使用的顺序是 bgr
p = 0.6

num_classes = len(pos_CLASSES)
attri_num_classes = len(attri_CLASSSES)
attri_num_classes_license_plate = len(attri_CLASSSES_license_plate)

weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

DetNet_300 = {
    'feature_maps' : [38, 19, 10, 10, 10, 10],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 32, 32, 32],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    'aspect_ratios' : [[2, 3], [2, 3], [2, 3], [2, 3], [2,3], [2,3]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}

MobileNetV1_300 = {
    'feature_maps' : [19, 10, 5, 3, 2, 1],

    'min_dim' : 300,

    'steps' : [16, 30, 60, 100, 150, 300],

    'min_sizes' : [60, 105, 150, 195, 240, 285],

    'max_sizes' : [105, 150, 195, 240, 285, 315],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2,3]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}

RefineDet_320 = {
    'feature_maps' : [40,20,10,5],

    'min_dim' : 320,

    'steps' : [16, 30, 60, 100, 150, 300],

    'min_sizes' : [60, 120, 180, 240],

    'max_sizes' : [120, 180, 240, 300],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}

RefineDet_320_V2 = {
    'feature_maps' : [40,20,10,5,3,1],

    'min_dim' : 320,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}

VOC_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [20, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    'aspect_ratios' : [[2, 3], [2, 3], [2, 3], [2, 3], [2,3], [2,3]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'anchor_level_k' : [9,9,9,9,9,9]
}

# VOC_300 = {
#     'feature_maps' : [1],

#     'min_dim' : 300,

#     'steps' : [300],

#     'min_sizes' : [264],

#     'max_sizes' : [315],

#     'aspect_ratios' : [[2, 3]],

#     'variance' : [0.1, 0.2],

#     'clip' : True,

#     'anchor_level_k' : [9,9,9,9,9,9]
# }

M2DET_300 = {
    'feature_maps' : [40, 20, 10, 5, 3, 1],

    'min_dim' : 320,

    'steps' : [8, 16, 32, 64, 107, 320],

    'min_sizes' : [25.6, 48.0, 105.6, 163.2, 220.8, 278.4],

    'max_sizes' : [48.0, 105.6, 163.2, 220.8, 278.4, 336.0],

    'aspect_ratios' : [[2, 3], [2, 3], [2, 3], [2, 3], [2,3], [2,3]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'anchor_level_k' : [3,3,3,3,3,3]
}

'''
VOC_512= {
    'feature_maps' : [64, 32, 16, 8, 4, 2, 1],

    'min_dim' : 512,

    'steps' : [8, 16, 32, 64, 128, 256, 512],

    'min_sizes'  : [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8 ],

    'max_sizes'  : [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2,3], [2,3]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}
'''
VOC_512= {
    'feature_maps' : [32, 16, 8, 4, 2, 1],

    'min_dim' : 512,

    'steps' : [16, 32, 64, 128, 256, 512],

    'min_sizes'  : [76.8, 153.6, 230.4, 307.2, 384.0, 460.8 ],

    'max_sizes'  : [153.6, 230.4, 307.2, 384.0, 460.8, 537.6],

    'aspect_ratios' : [[2, 3], [2, 3], [2, 3], [2,3], [2,3], [2,3]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}

COCO_300 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [21, 45, 99, 153, 207, 261],

    'max_sizes' : [45, 99, 153, 207, 261, 315],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}

COCO_512= {
    'feature_maps' : [64, 32, 16, 8, 4, 2, 1],

    'min_dim' : 512,

    'steps' : [8, 16, 32, 64, 128, 256, 512],

    'min_sizes' : [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],

    'max_sizes' : [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}

COCO_mobile_300 = {
    'feature_maps' : [19, 10, 5, 3, 2, 1],

    'min_dim' : 300,

    'steps' : [16, 32, 64, 100, 150, 300],

    'min_sizes' : [45, 90, 135, 180, 225, 270],

    'max_sizes' : [90, 135, 180, 225, 270, 315],

    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,
}
