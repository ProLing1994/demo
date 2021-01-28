from collections import OrderedDict
import numpy as np
import os 
import sys
import torch
import torch.nn as nn

caffe_root = '/home/huanyuan/code/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe 


ResNetDict = {
    "module.resnet.conv1.weight": "conv1,0",
    "module.resnet.bn1.weight": "scale_conv1,0",
    "module.resnet.bn1.bias": "scale_conv1,1",
    "module.resnet.bn1.running_mean": "bn_conv1,0",
    "module.resnet.bn1.running_var": "bn_conv1,1",
    "module.resnet.bn1.num_batches_tracked": "bn_conv1,2",
    "module.resnet.layer1.0.conv1.weight": "res2a_branch2a,0",
    "module.resnet.layer1.0.bn1.weight": "scale2a_branch2a,0",
    "module.resnet.layer1.0.bn1.bias": "scale2a_branch2a,1",
    "module.resnet.layer1.0.bn1.running_mean": "bn2a_branch2a,0",
    "module.resnet.layer1.0.bn1.running_var": "bn2a_branch2a,1",
    "module.resnet.layer1.0.bn1.num_batches_tracked": "bn2a_branch2a,2",
    "module.resnet.layer1.0.conv2.weight": "res2a_branch2b,0",
    "module.resnet.layer1.0.bn2.weight": "scale2a_branch2b,0",
    "module.resnet.layer1.0.bn2.bias": "scale2a_branch2b,1",
    "module.resnet.layer1.0.bn2.running_mean": "bn2a_branch2b,0",
    "module.resnet.layer1.0.bn2.running_var": "bn2a_branch2b,1",
    "module.resnet.layer1.0.bn2.num_batches_tracked": "bn2a_branch2b,2",
    "module.resnet.layer1.0.downsample.0.weight": "res2a_branch1,0",
    "module.resnet.layer1.0.downsample.1.weight": "scale2a_branch1,0",
    "module.resnet.layer1.0.downsample.1.bias": "scale2a_branch1,1",
    "module.resnet.layer1.0.downsample.1.running_mean": "bn2a_branch1,0",
    "module.resnet.layer1.0.downsample.1.running_var": "bn2a_branch1,1",
    "module.resnet.layer1.0.downsample.1.num_batches_tracked": "bn2a_branch1,2",
    "module.resnet.layer1.1.conv1.weight": "res2b_branch2a,0",
    "module.resnet.layer1.1.bn1.weight": "scale2b_branch2a,0",
    "module.resnet.layer1.1.bn1.bias": "scale2b_branch2a,1",
    "module.resnet.layer1.1.bn1.running_mean": "bn2b_branch2a,0",
    "module.resnet.layer1.1.bn1.running_var": "bn2b_branch2a,1",
    "module.resnet.layer1.1.bn1.num_batches_tracked": "bn2b_branch2a,2",
    "module.resnet.layer1.1.conv2.weight": "res2b_branch2b,0",
    "module.resnet.layer1.1.bn2.weight": "scale2b_branch2b,0",
    "module.resnet.layer1.1.bn2.bias": "scale2b_branch2b,1",
    "module.resnet.layer1.1.bn2.running_mean": "bn2b_branch2b,0",
    "module.resnet.layer1.1.bn2.running_var": "bn2b_branch2b,1",
    "module.resnet.layer1.1.bn2.num_batches_tracked": "bn2b_branch2b,2",
    "module.resnet.layer2.0.conv1.weight": "res3a_branch2a,0",
    "module.resnet.layer2.0.bn1.weight": "scale3a_branch2a,0",
    "module.resnet.layer2.0.bn1.bias": "scale3a_branch2a,1",
    "module.resnet.layer2.0.bn1.running_mean": "bn3a_branch2a,0",
    "module.resnet.layer2.0.bn1.running_var": "bn3a_branch2a,1",
    "module.resnet.layer2.0.bn1.num_batches_tracked": "bn3a_branch2a,2",
    "module.resnet.layer2.0.conv2.weight": "res3a_branch2b,0",
    "module.resnet.layer2.0.bn2.weight": "scale3a_branch2b,0",
    "module.resnet.layer2.0.bn2.bias": "scale3a_branch2b,1",
    "module.resnet.layer2.0.bn2.running_mean": "bn3a_branch2b,0",
    "module.resnet.layer2.0.bn2.running_var": "bn3a_branch2b,1",
    "module.resnet.layer2.0.bn2.num_batches_tracked": "bn3a_branch2b,2",
    "module.resnet.layer2.0.downsample.0.weight": "res3a_branch1,0",
    "module.resnet.layer2.0.downsample.1.weight": "scale3a_branch1,0",
    "module.resnet.layer2.0.downsample.1.bias": "scale3a_branch1,1",
    "module.resnet.layer2.0.downsample.1.running_mean": "bn3a_branch1,0",
    "module.resnet.layer2.0.downsample.1.running_var": "bn3a_branch1,1",
    "module.resnet.layer2.0.downsample.1.num_batches_tracked": "bn3a_branch1,2",
    "module.resnet.layer2.1.conv1.weight": "res3b_branch2a,0",
    "module.resnet.layer2.1.bn1.weight": "scale3b_branch2a,0",
    "module.resnet.layer2.1.bn1.bias": "scale3b_branch2a,1",
    "module.resnet.layer2.1.bn1.running_mean": "bn3b_branch2a,0",
    "module.resnet.layer2.1.bn1.running_var": "bn3b_branch2a,1",
    "module.resnet.layer2.1.bn1.num_batches_tracked": "bn3b_branch2a,2",
    "module.resnet.layer2.1.conv2.weight": "res3b_branch2b,0",
    "module.resnet.layer2.1.bn2.weight": "scale3b_branch2b,0",
    "module.resnet.layer2.1.bn2.bias": "scale3b_branch2b,1",
    "module.resnet.layer2.1.bn2.running_mean": "bn3b_branch2b,0",
    "module.resnet.layer2.1.bn2.running_var": "bn3b_branch2b,1",
    "module.resnet.layer2.1.bn2.num_batches_tracked": "bn3b_branch2b,2",
    "module.resnet.layer3.0.conv1.weight": "res4a_branch2a,0",
    "module.resnet.layer3.0.bn1.weight": "scale4a_branch2a,0",
    "module.resnet.layer3.0.bn1.bias": "scale4a_branch2a,1",
    "module.resnet.layer3.0.bn1.running_mean": "bn4a_branch2a,0",
    "module.resnet.layer3.0.bn1.running_var": "bn4a_branch2a,1",
    "module.resnet.layer3.0.bn1.num_batches_tracked": "bn4a_branch2a,2",
    "module.resnet.layer3.0.conv2.weight": "res4a_branch2b,0",
    "module.resnet.layer3.0.bn2.weight": "scale4a_branch2b,0",
    "module.resnet.layer3.0.bn2.bias": "scale4a_branch2b,1",
    "module.resnet.layer3.0.bn2.running_mean": "bn4a_branch2b,0",
    "module.resnet.layer3.0.bn2.running_var": "bn4a_branch2b,1",
    "module.resnet.layer3.0.bn2.num_batches_tracked": "bn4a_branch2b,2",
    "module.resnet.layer3.0.downsample.0.weight": "res4a_branch1,0",
    "module.resnet.layer3.0.downsample.1.weight": "scale4a_branch1,0",
    "module.resnet.layer3.0.downsample.1.bias": "scale4a_branch1,1",
    "module.resnet.layer3.0.downsample.1.running_mean": "bn4a_branch1,0",
    "module.resnet.layer3.0.downsample.1.running_var": "bn4a_branch1,1",
    "module.resnet.layer3.0.downsample.1.num_batches_tracked": "bn4a_branch1,2",
    "module.resnet.layer3.1.conv1.weight": "res4b_branch2a,0",
    "module.resnet.layer3.1.bn1.weight": "scale4b_branch2a,0",
    "module.resnet.layer3.1.bn1.bias": "scale4b_branch2a,1",
    "module.resnet.layer3.1.bn1.running_mean": "bn4b_branch2a,0",
    "module.resnet.layer3.1.bn1.running_var": "bn4b_branch2a,1",
    "module.resnet.layer3.1.bn1.num_batches_tracked": "bn4b_branch2a,2",
    "module.resnet.layer3.1.conv2.weight": "res4b_branch2b,0",
    "module.resnet.layer3.1.bn2.weight": "scale4b_branch2b,0",
    "module.resnet.layer3.1.bn2.bias": "scale4b_branch2b,1",
    "module.resnet.layer3.1.bn2.running_mean": "bn4b_branch2b,0",
    "module.resnet.layer3.1.bn2.running_var": "bn4b_branch2b,1",
    "module.resnet.layer3.1.bn2.num_batches_tracked": "bn4b_branch2b,2",
    "module.resnet.layer4.0.conv1.weight": "res5a_branch2a,0",
    "module.resnet.layer4.0.bn1.weight": "scale5a_branch2a,0",
    "module.resnet.layer4.0.bn1.bias": "scale5a_branch2a,1",
    "module.resnet.layer4.0.bn1.running_mean": "bn5a_branch2a,0",
    "module.resnet.layer4.0.bn1.running_var": "bn5a_branch2a,1",
    "module.resnet.layer4.0.bn1.num_batches_tracked": "bn5a_branch2a,2",
    "module.resnet.layer4.0.conv2.weight": "res5a_branch2b,0",
    "module.resnet.layer4.0.bn2.weight": "scale5a_branch2b,0",
    "module.resnet.layer4.0.bn2.bias": "scale5a_branch2b,1",
    "module.resnet.layer4.0.bn2.running_mean": "bn5a_branch2b,0",
    "module.resnet.layer4.0.bn2.running_var": "bn5a_branch2b,1",
    "module.resnet.layer4.0.bn2.num_batches_tracked": "bn5a_branch2b,2",
    "module.resnet.layer4.0.downsample.0.weight": "res5a_branch1,0",
    "module.resnet.layer4.0.downsample.1.weight": "scale5a_branch1,0",
    "module.resnet.layer4.0.downsample.1.bias": "scale5a_branch1,1",
    "module.resnet.layer4.0.downsample.1.running_mean": "bn5a_branch1,0",
    "module.resnet.layer4.0.downsample.1.running_var": "bn5a_branch1,1",
    "module.resnet.layer4.0.downsample.1.num_batches_tracked": "bn5a_branch1,2",
    "module.resnet.layer4.1.conv1.weight": "res5b_branch2a,0",
    "module.resnet.layer4.1.bn1.weight": "scale5b_branch2a,0",
    "module.resnet.layer4.1.bn1.bias": "scale5b_branch2a,1",
    "module.resnet.layer4.1.bn1.running_mean": "bn5b_branch2a,0",
    "module.resnet.layer4.1.bn1.running_var": "bn5b_branch2a,1",
    "module.resnet.layer4.1.bn1.num_batches_tracked": "bn5b_branch2a,2",
    "module.resnet.layer4.1.conv2.weight": "res5b_branch2b,0",
    "module.resnet.layer4.1.bn2.weight": "scale5b_branch2b,0",
    "module.resnet.layer4.1.bn2.bias": "scale5b_branch2b,1",
    "module.resnet.layer4.1.bn2.running_mean": "bn5b_branch2b,0",
    "module.resnet.layer4.1.bn2.running_var": "bn5b_branch2b,1",
    "module.resnet.layer4.1.bn2.num_batches_tracked": "bn5b_branch2b,2",
}

keypointDict = {
    "module.global_net.laterals.0.0.weight": "res5b_global,0",
    "module.global_net.laterals.0.1.weight": "scale5b_global,0",
    "module.global_net.laterals.0.1.bias": "scale5b_global,1",
    "module.global_net.laterals.0.1.running_mean": "bn5b_global,0",
    "module.global_net.laterals.0.1.running_var": "bn5b_global,1",
    "module.global_net.laterals.0.1.num_batches_tracked": "bn5b_global,2",
    "module.global_net.laterals.1.0.weight": "res4b_global,0",
    "module.global_net.laterals.1.1.weight": "scale4b_global,0",
    "module.global_net.laterals.1.1.bias": "scale4b_global,1",
    "module.global_net.laterals.1.1.running_mean": "bn4b_global,0",
    "module.global_net.laterals.1.1.running_var": "bn4b_global,1",
    "module.global_net.laterals.1.1.num_batches_tracked": "bn4b_global,2",
    "module.global_net.laterals.2.0.weight": "res3b_global,0",
    "module.global_net.laterals.2.1.weight": "scale3b_global,0",
    "module.global_net.laterals.2.1.bias": "scale3b_global,1",
    "module.global_net.laterals.2.1.running_mean": "bn3b_global,0",
    "module.global_net.laterals.2.1.running_var": "bn3b_global,1",
    "module.global_net.laterals.2.1.num_batches_tracked": "bn3b_global,2",
    "module.global_net.laterals.3.0.weight": "res2b_global,0",
    "module.global_net.laterals.3.1.weight": "scale2b_global,0",
    "module.global_net.laterals.3.1.bias": "scale2b_global,1",
    "module.global_net.laterals.3.1.running_mean": "bn2b_global,0",
    "module.global_net.laterals.3.1.running_var": "bn2b_global,1",
    "module.global_net.laterals.3.1.num_batches_tracked": "bn2b_global,2",
    "module.global_net.deconv.0.0.weight": "res5b_deconv,0",
    "module.global_net.deconv.0.1.weight": "scale5b_deconv,0",
    "module.global_net.deconv.0.1.bias": "scale5b_deconv,1",
    "module.global_net.deconv.0.1.running_mean": "bn5b_deconv,0",
    "module.global_net.deconv.0.1.running_var": "bn5b_deconv,1",
    "module.global_net.deconv.0.1.num_batches_tracked": "bn5b_deconv,2",
    "module.global_net.deconv.1.0.weight": "res4b_deconv,0",
    "module.global_net.deconv.1.1.weight": "scale4b_deconv,0",
    "module.global_net.deconv.1.1.bias": "scale4b_deconv,1",
    "module.global_net.deconv.1.1.running_mean": "bn4b_deconv,0",
    "module.global_net.deconv.1.1.running_var": "bn4b_deconv,1",
    "module.global_net.deconv.1.1.num_batches_tracked": "bn4b_deconv,2",
    "module.global_net.deconv.2.0.weight": "res3b_deconv,0",
    "module.global_net.deconv.2.1.weight": "scale3b_deconv,0",
    "module.global_net.deconv.2.1.bias": "scale3b_deconv,1",
    "module.global_net.deconv.2.1.running_mean": "bn3b_deconv,0",
    "module.global_net.deconv.2.1.running_var": "bn3b_deconv,1",
    "module.global_net.deconv.2.1.num_batches_tracked": "bn3b_deconv,2",
    "module.refine_net.cascade.0.0.conv1.weight": "res3b_cas1_1,0",
    "module.refine_net.cascade.0.0.bn1.weight": "res3b_cas1_1_Scale,0",
    "module.refine_net.cascade.0.0.bn1.bias": "res3b_cas1_1_Scale,1",
    "module.refine_net.cascade.0.0.bn1.running_mean": "res3b_cas1_1_BN,0",
    "module.refine_net.cascade.0.0.bn1.running_var": "res3b_cas1_1_BN,1",
    "module.refine_net.cascade.0.0.bn1.num_batches_tracked": "res3b_cas1_1_BN,2",
    "module.refine_net.cascade.0.0.conv2.weight": "res3b_cas1_2,0",
    "module.refine_net.cascade.0.0.bn2.weight": "res3b_cas1_2_Scale,0",
    "module.refine_net.cascade.0.0.bn2.bias": "res3b_cas1_2_Scale,1",
    "module.refine_net.cascade.0.0.bn2.running_mean": "res3b_cas1_2_BN,0",
    "module.refine_net.cascade.0.0.bn2.running_var": "res3b_cas1_2_BN,1",
    "module.refine_net.cascade.0.0.bn2.num_batches_tracked": "res3b_cas1_2_BN,2",
    "module.refine_net.cascade.0.0.conv3.weight": "res3b_cas1_3,0",
    "module.refine_net.cascade.0.0.bn3.weight": "res3b_cas1_3_Scale,0",
    "module.refine_net.cascade.0.0.bn3.bias": "res3b_cas1_3_Scale,1",
    "module.refine_net.cascade.0.0.bn3.running_mean": "res3b_cas1_3_BN,0",
    "module.refine_net.cascade.0.0.bn3.running_var": "res3b_cas1_3_BN,1",
    "module.refine_net.cascade.0.0.bn3.num_batches_tracked": "res3b_cas1_3_BN,2",
    "module.refine_net.cascade.0.0.downsample.0.weight": "res3b_cas1_ds,0",
    "module.refine_net.cascade.0.0.downsample.1.weight": "res3b_cas1_ds_Scale,0",
    "module.refine_net.cascade.0.0.downsample.1.bias": "res3b_cas1_ds_Scale,1",
    "module.refine_net.cascade.0.0.downsample.1.running_mean": "res3b_cas1_ds_BN,0",
    "module.refine_net.cascade.0.0.downsample.1.running_var": "res3b_cas1_ds_BN,1",
    "module.refine_net.cascade.0.0.downsample.1.num_batches_tracked": "res3b_cas1_ds_BN,2",
    "module.refine_net.cascade.0.1.weight": "refine_deconv,0",
    "module.refine_net.cascade.0.2.weight": "scale_refine_deconv,0",
    "module.refine_net.cascade.0.2.bias": "scale_refine_deconv,1",
    "module.refine_net.cascade.0.2.running_mean": "bn_refine_deconv,0",
    "module.refine_net.cascade.0.2.running_var": "bn_refine_deconv,1",
    "module.refine_net.cascade.0.2.num_batches_tracked": "bn_refine_deconv,2",
    "module.refine_net.final_predict.0.weight": "refine_dilate_1,0",
    "module.refine_net.final_predict.2.weight": "refine_dilate_2,0",
    "module.refine_net.final_predict.4.conv1.weight": "refine_out_1,0",
    "module.refine_net.final_predict.4.bn1.weight": "refine_out_1_Scale,0",
    "module.refine_net.final_predict.4.bn1.bias": "refine_out_1_Scale,1",
    "module.refine_net.final_predict.4.bn1.running_mean": "refine_out_1_BN,0",
    "module.refine_net.final_predict.4.bn1.running_var": "refine_out_1_BN,1",
    "module.refine_net.final_predict.4.bn1.num_batches_tracked": "refine_out_1_BN,2",
    "module.refine_net.final_predict.4.conv2.weight": "refine_out_2,0",
    "module.refine_net.final_predict.4.bn2.weight": "refine_out_2_Scale,0",
    "module.refine_net.final_predict.4.bn2.bias": "refine_out_2_Scale,1",
    "module.refine_net.final_predict.4.bn2.running_mean": "refine_out_2_BN,0",
    "module.refine_net.final_predict.4.bn2.running_var": "refine_out_2_BN,1",
    "module.refine_net.final_predict.4.bn2.num_batches_tracked": "refine_out_2_BN,2",
    "module.refine_net.final_predict.4.conv3.weight": "refine_out_3,0",
    "module.refine_net.final_predict.4.bn3.weight": "refine_out_3_Scale,0",
    "module.refine_net.final_predict.4.bn3.bias": "refine_out_3_Scale,1",
    "module.refine_net.final_predict.4.bn3.running_mean": "refine_out_3_BN,0",
    "module.refine_net.final_predict.4.bn3.running_var": "refine_out_3_BN,1",
    "module.refine_net.final_predict.4.bn3.num_batches_tracked": "refine_out_3_BN,2",
    "module.refine_net.final_predict.4.downsample.0.weight": "refine_concat_ds,0",
    "module.refine_net.final_predict.4.downsample.1.weight": "refine_concat_ds_Scale,0",
    "module.refine_net.final_predict.4.downsample.1.bias": "refine_concat_ds_Scale,1",
    "module.refine_net.final_predict.4.downsample.1.running_mean": "refine_concat_ds_BN,0",
    "module.refine_net.final_predict.4.downsample.1.running_var": "refine_concat_ds_BN,1",
    "module.refine_net.final_predict.4.downsample.1.num_batches_tracked": "refine_concat_ds_BN,2",
    "module.refine_net.final_predict.5.weight": "refine_out,0",
    "module.refine_net.final_predict.6.weight": "refine_out_Scale,0",
    "module.refine_net.final_predict.6.bias": "refine_out_Scale,1",
    "module.refine_net.final_predict.6.running_mean": "refine_out_BN,0",
    "module.refine_net.final_predict.6.running_var": "refine_out_BN,1",
    "module.refine_net.final_predict.6.num_batches_tracked": "refine_out_BN,2",
}


KWSDict = {
    "module.conv1.weight": "conv6,0",
    "module.conv1.bias": "conv6,1",
    "module.conv2.weight": "conv7,0",
    "module.conv2.bias": "conv7,1",
    "module.bn1.weight": "scale_conv6,0",
    "module.bn1.bias": "scale_conv6,1",
    "module.bn1.running_mean": "bn_conv6,0",
    "module.bn1.running_var": "bn_conv6,1",
    "module.bn1.num_batches_tracked": "bn_conv6,2",

}


ASRDict = {
    "module.conv1.weight": "dilation_conv1,0",
    "module.conv1.bias": "dilation_conv1,1",
    "module.conv2.weight": "dilation_conv2,0",
    "module.conv2.bias": "dilation_conv2,1",
    "module.conv3.weight": "conv_out,0",
    "module.conv3.bias": "conv_out,1",
    "module.bn1.weight": "scale_dilation_conv1,0",
    "module.bn1.bias": "scale_dilation_conv1,1",
    "module.bn1.running_mean": "bn_dilation_conv1,0",
    "module.bn1.running_var": "bn_dilation_conv1,1",
    "module.bn1.num_batches_tracked": "bn_dilation_conv1,2",
    "module.bn2.weight": "scale_dilation_conv2,0",
    "module.bn2.bias": "scale_dilation_conv2,1",
    "module.bn2.running_mean": "bn_dilation_conv2,0",
    "module.bn2.running_var": "bn_dilation_conv2,1",
    "module.bn2.num_batches_tracked": "bn_dilation_conv2,2",
}

LSTMDict = {
    "module.lstm.weight_ih_l0": "lstm1,0",
    "module.lstm.bias_ih_l0": "lstm1,1",
    "module.lstm.weight_hh_l0": "lstm1,2",
    "module.lstm.bias_hh_l0": "lstm1,1",
    "module.lstm.weight_ih_l1": "lstm2,0",
    "module.lstm.bias_ih_l1": "lstm2,1",
    "module.lstm.weight_hh_l1": "lstm2,2",
    "module.lstm.bias_hh_l1": "lstm2,1",
    "module.lstm.weight_ih_l2": "lstm3,0",
    "module.lstm.bias_ih_l2": "lstm3,1",
    "module.lstm.weight_hh_l2": "lstm3,2",
    "module.lstm.bias_hh_l2": "lstm3,1",
    "module.fc_out.weight": "fc_out,0",
    "module.fc_out.bias": "fc_out,1",
}


gestureDict = {
    "module.dilation_conv.weight": "dilation_conv,0",
    "module.fc_out.weight": "fc_out,0",
    "module.fc_out.bias": "fc_out,1",
}


if __name__ == '__main__':
    # input 
    model_path =  "/mnt/huanyuan/model/kws_model/RNN_language_model"
    pytorch_model = "cn_without_tone_lm_26.pth"
    caffe_prototxt = "cn_without_tone_lm_26.prototxt"
    caffe_model = "cn_without_tone_lm_26.caffemodel"

    # load pytorch model
    checkpoint = torch.load(os.path.join(model_path, pytorch_model), map_location=torch.device('cpu'))
    pytorch_network = checkpoint['state_dict']

    # torch_file, show model param
    torch_file=open(os.path.join(model_path, 'torch_file.txt'), 'w+', encoding='utf-8')
    for key, v in enumerate(pytorch_network):
        tmp_s = '"' + v + '":"",'
        torch_file.write(tmp_s + "\n")
        print(v)
    torch_file.close()

    # load caffe prototxt
    net = caffe.Net(os.path.join(model_path, caffe_prototxt), caffe.TEST)
    caffe_params_dict = net.params
    print("[Information:] Caffe params: ", len(caffe_params_dict), type(caffe_params_dict), caffe_params_dict)

    # load dict 
    pytorch_to_caffe_dict=dict(LSTMDict)
    # nameDict=dict(ResNetDict, **KWSDict, **LSTMDict)
    pytorch_layer_name_list = list(pytorch_to_caffe_dict.keys())
    caffe_layer_name_list = list(pytorch_to_caffe_dict.values())

    # transform
    for key, pytorch_network_tensor in enumerate(pytorch_network):
    #for pytorch_network_tensor in network.state_dict():   change happend here
        pytorch_layer_name = "module." + pytorch_network_tensor
        # pytorchLayerName = pytorch_network_tensor

        if pytorch_layer_name not in pytorch_layer_name_list:
            print("[ERROR:] Pytorch tensor %s is not in nameDict\n"%pytorch_layer_name)
            continue
            sys.exit()

        pytorch_layer_param = pytorch_network[pytorch_network_tensor]
        #pytorch_layer_param = network.state_dict()[param_tensor]   change happend here
        caffe_layer_param = pytorch_to_caffe_dict[pytorch_layer_name]
        print("[Information:] Pytorch params: ", pytorch_network_tensor, pytorch_to_caffe_dict[pytorch_layer_name])

        if "," in caffe_layer_param:
            caffe_layer_name, caffe_layer_mat_num = caffe_layer_param.strip().split(",")
            caffe_layer_mat_num = int(caffe_layer_mat_num)
            if caffe_layer_name not in caffe_params_dict:
                print("[ERROR:] caffe_layer_name is not in caffe prototxt")
            print("[Information:] Caffe params: ",caffe_layer_name, caffe_layer_mat_num)

            if "num_batches_tracked" in pytorch_network_tensor:
                caffe_params_dict[caffe_layer_name][caffe_layer_mat_num].data[...] = np.array([1.0])
            elif "lstm.weight" in pytorch_network_tensor:
                # 注意：pytorch model & caffe model，lstm 装载数据类型方式不同，pytorch：`(W_ii|W_if|W_ig|W_io)` caffe：`(W_ii|W_if|W_io|W_ig)`
                param_pytorch = pytorch_layer_param.cpu().data.numpy()
                param_caffe = param_pytorch.copy()
                # param_caffe[768:] = param_pytorch[512:768]
                # param_caffe[512:768] = param_pytorch[768:]
                param_caffe[int(param_pytorch.shape[0]/4*3): ] = param_pytorch[int(param_pytorch.shape[0]/4*2): int(param_pytorch.shape[0]/4*3)]
                param_caffe[int(param_pytorch.shape[0]/4*2): int(param_pytorch.shape[0]/4*3)] = param_pytorch[int(param_pytorch.shape[0]/4*3):]

                assert caffe_params_dict[caffe_layer_name][caffe_layer_mat_num].data.shape == param_pytorch.shape
                caffe_params_dict[caffe_layer_name][caffe_layer_mat_num].data[...] = param_caffe
            elif "lstm.bias" in pytorch_network_tensor:
                param_pytorch = pytorch_layer_param.cpu().data.numpy()
                param_caffe = param_pytorch.copy()
                # param_caffe[768:] = param_pytorch[512:768]
                # param_caffe[512:768] = param_pytorch[768:]
                param_caffe[int(param_pytorch.shape[0]/4*3): ] = param_pytorch[int(param_pytorch.shape[0]/4*2): int(param_pytorch.shape[0]/4*3)]
                param_caffe[int(param_pytorch.shape[0]/4*2): int(param_pytorch.shape[0]/4*3)] = param_pytorch[int(param_pytorch.shape[0]/4*3):]

                assert caffe_params_dict[caffe_layer_name][caffe_layer_mat_num].data.shape == param_pytorch.shape
                caffe_params_dict[caffe_layer_name][caffe_layer_mat_num].data[...] += param_caffe
            else:
                caffe_params_dict[caffe_layer_name][caffe_layer_mat_num].data[...] = pytorch_layer_param.cpu().data.numpy()

    net.save(os.path.join(model_path, caffe_model))
    print("net save end")
    sys.exit()
