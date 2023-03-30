#!/usr/bin/env bash
# ```
# # 保存 deploy 模型
# # yolov6(RepVGGBlock) -> yolov6_deploy 生成 deploy 模型（使用 deploy == False 的 config）
# /home/huanyuan/code/demo/Image/detection2d/mmdetection/demo/detector/yolov6_deploy.py

# # 保存 caffe 模型
# # deploy 模型转换为 caffemodel（使用 deploy == TRUE 的 config）
# /home/huanyuan/code/demo/Image/detection2d/mmdet2caffe/mmdet2caffe.sh 
# ```

# CONFIG=/mnt/huanyuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0824/yolov6_zg_bmx_deploy.py
# CHECKPOINT=/mnt/huanyuan/model/image/yolov6/yolov6_zg_bmx_adas_bsd_zg_data_0824/epoch_300_deploy.pth
# NAME=yolov6_zg_bmx_deploy_0824
# OUTPUT_DIR=/home/huanyuan/code/demo/Image/detection2d/mmdet2caffe/caffe_model

# CONFIG=/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0327/yolov6_rm_c28_deploy.py
# CHECKPOINT="/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0327/epoch_1_deploy.pth"
# NAME=yolov6_rm_c28
# OUTPUT_DIR=/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0327

CONFIG=/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/yolov6_rm_c28_deploy.py
CHECKPOINT=/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/epoch_340_deploy.pth
NAME=yolov6_rm_c28
OUTPUT_DIR=/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320

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