 #!/usr/bin/env bash
CONFIG=/home/huanyuan/code/demo/Image/2ddetection/mmdetection/configs/zhiguan/detect/yoloxv2_rmvgg_rmfpn_nova_3level_zhiguan_non_car_person_detection_320x320.py
CHECKPOINT=/home/huanyuan/code/demo/Image/2ddetection/mmdetection/checkpoints/epoch_24.pth
NAME=yoloxv2_guoshengdaoData_patch_mix
OUTPUT_DIR=/home/huanyuan/code/demo/Image/2ddetection/mmdet2caffe/caffe_model

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