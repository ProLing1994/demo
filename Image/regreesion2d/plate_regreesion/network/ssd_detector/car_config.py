from ssd_detector.model_ver_relu import MobileNetV1


class Config(object):
    def __init__(self):
        self.folder_name = "MobileNetSmallV1_Hunan"
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
        self.num_class = 3
        self.train_set_mat_file = "/mnt/ssd_disk_1/jhwen/data/car_regression_voc_bbox/train_crop.mat"
        self.validation_set_mat_file = "/mnt/ssd_disk_1/jhwen/data/car_regression_voc_bbox/val_crop.mat"
        self.log_flag = True
        self.print_freq = 5
        self.epochs = 50
        self.train_batch_size = 512
        self.validation_batch_size = 64
        self.train_num_workers = 32
        self.validation_num_workers = 8
        self.validation_images_number = -1
        self.color_mode = "bgr"
        self.width_mul = 1.
        self.learning_rate = 1e-3
        self.validation_steps = int(self.validation_images_number / self.validation_batch_size)
        # self.ckpt_path = "torch_model/MobileNetSmallV1_small_bgr_2019_09_03_13_27_47/ckpt/20.ckpt"
        # self.OHEM_epochs = 20
        # self.online_multi_datagen_epoch = 10
        # self.OMG_mat_file = ["out_hunan_2019_11_26_10_14_05.mat", "train_2019_09_03_18_48_16.mat"]
        # self.OMG_img_path = ["D:/data/CarHunan/", "D:/data/MTCNN_VOC/JPEGImages/"]
        # self.OMG_sample_rate = [0.2, 0.8]
