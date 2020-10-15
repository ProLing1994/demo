from easydict import EasyDict as edict

__C = edict()
cfg = __C

##################################
# general parameters
##################################

__C.general = {}

# the label
__C.general.positive_label = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
__C.general.negative_label = ["__silence__", "__unknown__"]
__C.general.num_classes = len(__C.general.positive_label) + len(__C.general.negative_label)

# test after save pytorch model
__C.general.is_test = False

# the output of training models and logging files
__C.general.save_dir = "/home/huanyuan/code/third_code/honk/model/test/"

# set certain epoch to continue training, set -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training
__C.general.num_gpus = 1

##################################
# data set parameters
##################################

__C.dataset = {}

# the number of input channel, currently only support 1 channel input.
__C.dataset.input_channel = 1

# input size of training data (w, h), unit: voxel
__C.dataset.data_size = [40, 98]

####################################
# training loss
####################################

__C.loss = {}

# the loss name, support 'softmax'
__C.loss.name = 'softmax'

#####################################
# net
#####################################

__C.net = {}

# the network name
__C.net.name = 'cnn-trad-pool2'

######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.num_epochs = 2001

# the number of samples in a batch
__C.train.batch_size = 64

# the number of threads for IO
__C.train.num_threads = 32

# the number of batches to update loss curve
__C.train.plot_snapshot = 100

# the number of epochs to save model
__C.train.save_epochs = 50


######################################
# learning rate parameters
######################################

# learning rate = lr*gamma**(epoch//step_size)
__C.train.lr = 1e-3

# step size for step learning rate
__C.train.lr_step_size = 0

# gamma for learning rate
__C.train.lr_gamma = 0.9


######################################
# optimizer parameters
######################################

# optimizer, support SGD and Adam
__C.train.optimizer = 'Adam'

# SGD, Adam momentum
__C.train.momentum = 0.9

# SGD,Adam weight decay
__C.train.weight_decay = 0.0

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)

######################################
# debug parameters
######################################

__C.debug = {}

# random seed used in training
__C.debug.seed = 0