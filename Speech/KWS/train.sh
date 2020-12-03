#!/bin/bash

# 该脚本进行模型训练

stage=2 

# init
config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py

echo "train.sh"

# prepare dataset
if [ $stage -le 1 ];then
    script/dataset/prepare_dataset.sh || exit 1
fi

# train model
if [ $stage -le 2 ];then
    python train.py -i $config_file || exit 1
fi

echo "train.sh succeeded"