 #!/usr/bin/env bash
# CONFIG=/mnt/huanyuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0722/yolov6_zg_bmx.py
# CHECKPOINT=/mnt/huanyuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0722/epoch_300.pth
# NAME=yolov6_zg_bmx_0722
# OUTPUT_DIR=/home/huanyuan/code/demo/Image/detection2d/mmdet2caffe/caffe_model

# CONFIG=/mnt/huanyuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0722/yolov6_zg_bmx_deploy.py
# CHECKPOINT=/mnt/huanyuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0722/epoch_300_deploy.pth
# NAME=yolov6_zg_bmx_deploy_0722
# OUTPUT_DIR=/home/huanyuan/code/demo/Image/detection2d/mmdet2caffe/caffe_model

CONFIG=/mnt/huanyuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_data_attribute_conv/yolov6_zg_bmx_attrtibute_conv.py
CHECKPOINT=/mnt/huanyuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_data_attribute_conv/epoch_5.pth
NAME=yolov6_zg_bmx_attribute_conv
OUTPUT_DIR=/home/huanyuan/code/demo/Image/detection2d/mmdet2caffe/caffe_model

python deployment/pytorch2caffe/mmdet2caffe.py \
   --config-file $CONFIG \
   --model-path $CHECKPOINT \
   --name $NAME \
   --output $OUTPUT_DIR

# example
# python tools/deployment/pytorch2caffe/mmdet2caffe.py \
#    --config-file configs/bsd/all_in_one/fcos_rmvgg_rmfpn_nova_with_FCNMaskHead_3level_segrmitopkloss_all_in_one_320x320.py \
#    --model-path /lmliu/lmliu/logs/mmdetection/bsd_all_in_one/fcos_rmvgg_rmfpn_nova_with_FCNMaskHead_3level_segrmitopkloss_all_in_one_320x320/epoch_12.pth \
#    --name fcos_rmvgg_rmfpn_nova_with_FCNMaskHead_320x320 \
#    --output weights/caffe_models/bsd_all_in_one/