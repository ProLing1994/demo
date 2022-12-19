import sys 

# sys.path.insert(0, '/home/huanyuan/code/demo/Image')
sys.path.insert(0, '/yuanhuan/code/demo/Image')
from regreesion2d.plate_regreesion.network.model import MobileNetV1

class Config(object):
    def __init__(self):
        self.folder_name = "MobileNetSmallV1_Turkey_Wdm_PT_sgd"
        self.save_path_prefix = "/jhwen/train/out/plate_regression/turkey"
        self.img_mean = 127.5
        self.img_scale = 127.5
        self.model = MobileNetV1
        self.input_channel = 32
        self.output_channel = 512
        self.interverted_residual_setting = [
            # c, n, s
            [32, 1, 1],
            [64, 2, 2],
            [128, 2, 2],
            [256, 6, 2],
            [512, 2, 2],
        ]
        self.num_class = 7
        self.plate_only = True
        self.train_set_mat_file = "/jhwen/train/data/plate_after_crop/turkey_regression/train_crop_test.mat"
        self.validation_set_mat_file = "/jhwen/train/data/plate_after_crop/turkey_regression/val_crop_test.mat"
        self.log_flag = True
        self.print_freq = 5
        self.epochs = 20
        self.weighted_mse_dis_loss = True
        self.loss_config = {"mse_loss_weight": [0, 0, 0, 1, 1, 1, 1], "mse_dis_loss_weight": [1, 0.3]}
        self.train_batch_size = 256
        self.validation_batch_size = 64
        self.train_num_workers = 32
        self.validation_num_workers = 4
        self.validation_images_number = -1
        self.color_mode = "bgr"
        self.width_mul = 1.
        self.learning_rate = 1e-4
        self.ckpt_path = "torch_model/MobileNetSmallV1_PO_singapore_Wdm_2020_03_19_17_36_30_PT_best.pt"
        self.validation_steps = int(self.validation_images_number/self.validation_batch_size)
        # self.OHEM_flag = True
        # self.OHEM_epochs = 40
        # self.freeze_epochs = 20
        self.warm_up_epochs = 5
        # self.loss_weight = [[0, 0, 0, 1, 1, 1, 1]]
        # self.smooth_l1_flag = False
        # self.distance_iou_loss_flag = True

# /jhwen/train/code/plate_regression/train.sh -c config/config_turkey.py
