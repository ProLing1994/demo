#!/bin/bash

stage=0

echo "script/ESC-50/run.sh"

config_file=/home/huanyuan/code/demo/Speech/SED/config/sed_config_ESC50.py

# train test dataset split 
# 功能：训练集、测试集划分
if [ $stage -le 1 ];then
	python create_train_test_indexes.py create_indexes -i $config_file || exit 1
fi

# preload data
# 功能：为加快模型训练速度，预先加载数据，为进一步加快模型训练速度，保存音频文件为lmdb格式文件
if [ $stage -le 2 ];then
	python data_preload_lmdb.py preload_audio_lmdb -i $config_file || exit 1
fi