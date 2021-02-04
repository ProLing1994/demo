#!/bin/bash

# 该脚本用于训练集、验证集、测试集的分配，以及数据的预先加载

# 注：执行脚本前，请执行以下操作：
# 1. 执行文件 prepare_dataset_{}.py 整理数据集目录结构，同时需要自定义函数 get_hash_name
# 2. 执行文件 data_clean.py 运用 VAD 工具对音频数据进行切割，仅保留唤醒词部分
# 3. 若步骤 2 效果不理想，则执行脚本，dataset_align/dataset_align.sh，运行 kaldi 强制对齐工具进行强制对齐，对音频数据进行切割，仅保留唤醒词部分
# 4. 自行检测数据的完整性和正确性

stage=6

# init
config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_speech.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_xiaoyu.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_xiaorui.py
# config_file=/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_pretrain.py

echo "script/dataset/prepare_dataset.sh"

# train test dataset split 
if [ $stage -le 1 ];then
	python data_train_test_split.py -i $config_file || exit 1
fi

# add mining difficult sample
# 如果配置文件 difficult_sample_mining = False，将不会添加到 total_data_files.csv
if [ $stage -le 2 ];then
	python ../mining_difficult_sample/data_train_test_split_mining_difficult_sample.py \
			--config_file $config_file \
			--difficult_sample_mining_dir /mnt/huanyuan/data/speech/kws/difficult_sample_mining/difficult_sample_mining_11122020/clean_audio/ || exit 1
	python ../mining_difficult_sample/data_train_test_split_mining_difficult_sample.py \
			--config_file $config_file \
			--difficult_sample_mining_dir /mnt/huanyuan/data/speech/kws/difficult_sample_mining/difficult_sample_mining_12212020/audio/ || exit 1
fi

# prepare align dataset, clean the dataset according to the alignment results
# 如果文件夹下不存在 kaldi_type，则不进行 update_dataset
if [ $stage -le 3 ];then
	python update_dataset.py --config_file $config_file || exit 1
fi

# speed volume augumentation
# 如果配置文件 speed_volume_on = False，将不会进行 augumentation
if [ $stage -le 4 ];then
	python ../dataset_augmentation/speed_volume_augumentation.py --config_file $config_file || exit 1
fi

# preload data
if [ $stage -le 5 ];then
	python data_preload_audio.py --config_file $config_file || exit 1
fi

# analysis dataset
if [ $stage -le 6 ];then
	python ../analysis_dataset/analysis_audio_length.py --config_file $config_file || exit 1
	python ../analysis_dataset/analysis_data_distribution.py --config_file $config_file || exit 1
fi

echo "script/dataset/prepare_dataset.sh succeeded"