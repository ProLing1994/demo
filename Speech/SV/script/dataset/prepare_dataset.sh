#!/bin/bash

# 该脚本用于训练集、验证集、测试集的分配，以及数据的预先加载

stage=2

# init
# config_file=/home/huanyuan/code/demo/Speech/SV/config/sv_config_english_TI_SV.py
config_file=/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TI_SV.py

echo "script/dataset/prepare_dataset.sh"

# vad 
# 功能：vad，数据清洗剔除静音音频部分
if [ $stage -le 1 ];then
	python data_vad.py -i $config_file || exit 1
fi

# train test dataset split 
# 功能：训练集、测试集划分
if [ $stage -le 2 ];then
	python data_train_test_split.py -i $config_file || exit 1
fi

# preload data
# 功能：为加快模型训练速度，预先加载数据
# 预先加载数据格式：为进一步加快模型训练速度，保存音频文件为 lmdb 格式文件
# 说明：脚本使用 librosa.core.load 加载音频文件，所需参：sample_rate，跟特征维度没有关系
if [ $stage -le 3 ];then
	python data_preload_audio_lmdb.py --config_file $config_file || exit 1
fi

echo "script/dataset/prepare_dataset.sh succeeded"