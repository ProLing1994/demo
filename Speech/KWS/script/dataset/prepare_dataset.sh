#!/bin/bash

# 该脚本用于训练集、验证集、测试集的分配，以及数据的预先加载

# 注：执行脚本前，请执行以下操作：
# 1. 执行文件 prepare_dataset_{}.py 整理数据集目录结构，同时需要自定义函数 get_hash_name
# 2. 执行文件 data_clean.py 运用 VAD 工具对音频数据进行切割，仅保留唤醒词部分
# 3. 若步骤 2 效果不理想，则执行脚本，dataset_align/dataset_align.sh，运行 kaldi 强制对齐工具进行强制对齐，对音频数据进行切割，仅保留唤醒词部分
# 4. 若步骤 2、步骤 3 效果不理想，则手动进行检测（目前英文关键词只能通过手动检查的方法）
# 4. 自行检测数据的完整性和正确性

stage=1

# init
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_speech.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui8k.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui16k.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_xiaoyu.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_xiaorui.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_pretrain.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_activatebwc.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_heybodycam.py
config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoan8k.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoan16k.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_nihaoxiaoan8k.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_nihaoxiaoan16k.py

echo "script/dataset/prepare_dataset.sh"

# train test dataset split 
# 功能：训练集、测试集划分
if [ $stage -le 1 ];then
	python data_train_test_split.py -i $config_file || exit 1
fi

# preload data
# 功能：为加快模型训练速度，预先加载数据
# 预先加载数据格式：为进一步加快模型训练速度，保存音频文件为lmdb格式文件
# 说明：脚本使用 librosa.core.load 加载音频文件，所需参：sample_rate、clip_duration_ms，跟特征维度没有关系
if [ $stage -le 2 ];then
	python data_preload_audio_lmdb.py --config_file $config_file || exit 1
fi

# analysis dataset
if [ $stage -le 3 ];then
	python ../analysis_dataset/analysis_audio_length.py --config_file $config_file || exit 1
	python ../analysis_dataset/analysis_data_distribution.py --config_file $config_file || exit 1
fi

echo "script/dataset/prepare_dataset.sh succeeded"