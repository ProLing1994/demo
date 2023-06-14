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

# CONFIG=/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/yolov6_rm_c28_deploy.py
# CHECKPOINT=/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320/epoch_340_deploy.pth
# NAME=yolov6_rm_c28
# OUTPUT_DIR=/mnt/huanyuan/model/image/yolov6/yolov6_c28_car_0320

# CONFIG=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_20230526/yolov6_face_wider_face_deploy.py
# CHECKPOINT=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_20230526/epoch_400_deploy.pth
# NAME=yolov6_face
# OUTPUT_DIR=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_20230526

# CONFIG=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_center_offset_20230525/yolov6_face_wider_face_deploy.py
# CHECKPOINT=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_center_offset_20230525/epoch_400_deploy.pth
# NAME=yolov6_face_center_offset
# OUTPUT_DIR=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_wider_face_center_offset_20230525

# CONFIG=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_qa_wider_face_20230605/yolov6_qa_face_wider_face_deploy.py
# CHECKPOINT=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_qa_wider_face_20230605/epoch_800_deploy.pth
# NAME=yolov6_face_landmark_qa_20230605_800
# OUTPUT_DIR=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_qa_wider_face_20230605

# CONFIG=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_degree_wider_face_20230607/yolov6_face_wider_face_deploy.py
# CHECKPOINT=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_degree_wider_face_20230607/epoch_300_deploy.pth
# NAME=yolov6_landmark_degree_wider_face_20230607
# OUTPUT_DIR=/mnt/huanyuan/model/image/yolov6/yolov6_landmark_degree_wider_face_20230607

CONFIG=/mnt/huanyuan/model/image/yolov6/yolox_landmark_wider_face_20230612/yolox_face_wider_face.py
CHECKPOINT=/mnt/huanyuan/model/image/yolov6/yolox_landmark_wider_face_20230612/epoch_300_deploy.pth
NAME=yolox_landmark_wider_face_20230612_deploy
OUTPUT_DIR=/mnt/huanyuan/model/image/yolov6/yolox_landmark_wider_face_20230612

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