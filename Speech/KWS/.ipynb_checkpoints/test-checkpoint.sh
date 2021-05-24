#!/bin/bash

# 该脚本进行模型测试

stage=1

# init
# # xiaorui
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py
config_file=/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_1_finetune_res15_12082020/kws_config_xiaorui.py
model_epoch=500

# xiaoyu
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py
# model_epoch=-1

# pretrain
# config_file=/mnt/huanyuan/model/model_10_30_25_21/model/kws_pretrain_12102020/kws_config_pretrain.py
# model_epoch=-1

echo "test.sh"

# test model
if [ $stage -le 1 ];then
    python ./infer/infer.py --input $config_file --epoch $model_epoch --result_mode "min" || exit 1
    python ./infer/infer.py --input $config_file --epoch $model_epoch --result_mode "average_duration_ms" || exit 1
fi

echo "test.sh succeeded"