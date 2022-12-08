#!/bin/bash

stage=1

# init
# data_dir=/mnt/huanyuan2/data/speech/asr/English/VCC2020-database/dataset/train/
# output_dir=/mnt/huanyuan/data/speech/vc/English_dataset/dataset_audio_hdf5/VCC2020/conf_test/
data_dir=/mnt/huanyuan2/data/speech/vc/Chinese/vc_test/train/
output_dir=/mnt/huanyuan/data/speech/vc/Chinese_dataset/dataset_audio_hdf5/BZNSYP_Aishell3/conf_test/

echo "gen_conf_api.sh"

# 绘图
if [ $stage -le 1 ];then
	python f0_npow_hisgram.py -d $data_dir -o $output_dir || exit 1
fi

# 生成 conf
if [ $stage -le 2 ];then
	python gen_conf_api.py -d $data_dir -o $output_dir || exit 1
fi

echo "gen_conf_api.sh succeeded"