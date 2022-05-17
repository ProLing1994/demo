
import cv2
import matplotlib.pyplot as plt
import numpy as np


import openvino.inference_engine as ie

rgb_means = (104, 117, 123)
jpg_path = "/home/huanyuan/temp/jpg/0000000000000000-220424-140011-140025-000001504000/0000000000000000-220424-140011-140025-000001504000_70.jpg"
modle_xml = "/mnt/huanyuan/model_final/image_model/ssd_rfb_zg/car_bus_truck_licenseplate_softmax_zg_2022-04-25-18/openvino_model/SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-04-25-18.xml"

def preprocess(src, W, H):
    img = cv2.resize(src, (W, H)).astype(np.float32)
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
    priors = out['mbox_priorbox'][0]
    boxes = decode(boxes, priors, [0.1, 0.2])
    boxes *= np.array([w, h, w, h])
    scores = out['mbox_conf_reshape'][0]
    attris = out['mbox_attri_reshape'][0]
    return (boxes.astype(np.int32), scores, attris)
    
# Inference Engine API
core = ie.IECore()

# Read a model from a drive
network = core.read_network(modle_xml)

# Load the Model to the Device
exec_network = core.load_network(network, "CPU")
input_blob = next(iter(network.input_info))

# Fill input tensors
infer_request = exec_network.requests[0]
# Get input blobs mapped to input layers names
input_blobs = infer_request.input_blobs
data = input_blobs["data"].buffer

# Text detection models expects image in BGR format
image = cv2.imread(jpg_path)

# N,C,H,W = batch size, number of channels, height, width
N, C, H, W = data.shape

# Resize image to meet network expected input sizes
image = preprocess(image, W, H)

# Reshape to network input shape
input_image = np.expand_dims(image.transpose(2, 0, 1), 0)

# Fill the first blob ...
# Start Inference
res = exec_network.infer(inputs={input_blob: input_image})
boxes = res['mbox_loc_reshape'][0]
priors = res['mbox_priorbox'][0]
scores = res['mbox_conf_reshape'][0]
attris = res['mbox_attri_reshape'][0]
print()