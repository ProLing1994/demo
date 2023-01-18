import torch
from models.LANENET import LANENET_NOVT, LANENET_BIG
from net import BuildNet
from pytorch2caffe import pytorch_to_caffe

if __name__ == "__main__":
    
    size = (1, 3, 128, 128)
    name = 'R151seg_wa'
    wpath = 'weights/base_220912_32c_128_lanenet_big/Seg_epoch219.pth'
    spath = 'weights_caffe/base_220912_32c_128_lanenet_big/'
    
    # model = BuildNet('Seg16', 5, 3, (128,128), 16, 'RFBResBlock', 1, softmax=True)
    model = LANENET_BIG(5, False, 32)
    model.load_state_dict(torch.load(wpath))
    model.eval()
    inputs = torch.randn(size)

    pytorch_to_caffe.trans_net(model, inputs, name)
    pytorch_to_caffe.save_prototxt("{}/{}.prototxt".format(spath, name))
    pytorch_to_caffe.save_caffemodel("{}/{}.caffemodel".format(spath, name))

    print(f"Export caffe model successfully!")



# 修改
"""
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 128
      dim: 128   }
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv_blob1"
  convolution_param {
    num_output: 32
    bias_term: false
    pad: 1
    kernel_size: 3
    group: 1
    stride: 2
    weight_filler {
      type: "xavier"
    }
    dilation: 1
  }
}



layer {
  name: "prob"
  type: "Softmax"
  bottom: "conv_transpose_blob3"
  top: "prob"
  softmax_param {
    axis: 1
  }
}
"""
