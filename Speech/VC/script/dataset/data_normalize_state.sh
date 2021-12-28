#!/bin/bash

stage=1

# init
config_file=/home/huanyuan/code/demo/Speech/VC/config/cycle_vae/vc_config_cycle_vae.py

echo "script/dataset/data_normalize_state.sh"

# preload data normalize state
# 功能：保存数据均值、方差，归一化数据分布
if [ $stage -le 1 ];then
	python data_normalize_state.py -i $config_file || exit 1
fi

echo "script/dataset/data_normalize_state.sh succeeded"