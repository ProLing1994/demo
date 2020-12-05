#!/bin/bash

# 该脚本运行 kaldi 强制对齐工具进行强制对齐
# 作用 1：后续用于语音切割
# 作用 2：后续用于基于帧对齐像素颗粒度的语音唤醒
# 注意：执行文本前，需要注意音频格式采样率为 16K，否则会报错

stage=1

# init
# xiaorui
# data_set=/mnt/huanyuan/data/speech/kws/xiaorui_dataset/original_dataset/XiaoRuiDataset_12022020
# keyword="xiaorui"
# keyword_chinese="小 锐 小 锐"
# keyword_chinese_output="小,锐,小#,锐#"

# xiaoyu
# data_set=/mnt/huanyuan/data/speech/kws/xiaoyu_dataset/original_dataset/XiaoYuDataset_11032020/
data_set=/mnt/huanyuan/data/speech/kws/xiaoyu_dataset/original_dataset/XiaoYuDataset_11192020/
keyword="xiaoyu"
keyword_chinese="小 鱼 小 鱼"
keyword_chinese_output="小,鱼,小#,鱼#"

echo "script/dataset_align/dataset_align.sh"

# kaldi type data preparation
# 注意：代码需要根据唤醒词进行相应的更改，spk/device/text
if [ $stage -le 1 ];then
	python prepare_dataset_kaldi_type.py --dataset_path $data_set --dest_path $data_set/kaldi_type --keyword_list $keyword --keyword_chinese_name_list "$keyword_chinese" || exit 1
fi

# align words time index
if [ $stage -le 2 ];then
    echo "[Begin] align_nnet3_word"
    cd ./src
	echo $PWD
	./align_nnet3_word.sh $data_set || exit 1
	cd ../
    echo "[Done] align_nnet3_word"
fi

# cut keyword data
if [ $stage -le 3 ];then
    python cut_keyword_wav.py --ctm_file $data_set/kaldi_type/tmp/nnet3_align/ctm --wav_file $data_set/kaldi_type/wav.scp --keyword_list "$keyword_chinese_output" --save_dir $data_set/kaldi_cut_keyword
fi

echo "script/dataset_align/dataset_align.sh succeeded"