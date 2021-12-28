#!/bin/bash

stage=1

# init
config_file=/home/huanyuan/code/demo/Speech/VC/config/cycle_vae/vc_config_cycle_vae.py

echo "script/dataset/data_train_test_split.sh"

# train test dataset split 
# 功能：训练集、测试集划分
if [ $stage -le 1 ];then
	python data_train_test_split.py -i $config_file || exit 1
fi

echo "script/dataset/data_train_test_split.sh succeeded"