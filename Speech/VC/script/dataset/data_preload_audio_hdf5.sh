#!/bin/bash

stage=1

# init
# config_file=/home/huanyuan/code/demo/Speech/VC/config/cyclevae/vc_config_cyclevae.py
config_file=/home/huanyuan/code/demo/Speech/VC/config/cyclevae/vc_config_chinese_cyclevae.py

echo "script/dataset/data_preload_audio_hdf5.sh"

# preload data
# 功能：预先加载数据
if [ $stage -le 1 ];then
	python data_preload_audio_hdf5.py -i $config_file || exit 1
fi

echo "script/dataset/data_preload_audio_hdf5.sh succeeded"