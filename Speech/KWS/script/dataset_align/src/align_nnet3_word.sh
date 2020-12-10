#!/bin/bash

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dataset_path=$1
nj=3
stage=0

if [ $stage -le 0 ]; then

utils/fix_data_dir.sh $dataset_path || exit 1;

steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj $dataset_path $dataset_path/tmp/log/make_mfcc/ $dataset_path/tmp/mfcc_hires || exit 1

utils/fix_data_dir.sh $dataset_path || exit 1

steps/compute_cmvn_stats.sh $dataset_path $dataset_path/tmp/log/make_mfcc $dataset_path/tmp/cmvn || exit 1

utils/fix_data_dir.sh $dataset_path || exit 1;


fi

if [ $stage -le 1 ]; then

steps/nnet3/align.sh --nj $nj --cmd "$train_cmd" --use-gpu false --scale_opts "--transition-scale=1.0 --acoustic-scale=10.0 --self-loop-scale=0.1" \
                  $dataset_path model/lang_aishell model/nnet3/tdnn_sp/ $dataset_path/tmp/nnet3_align || exit 1

fi


if [ $stage -le 2 ]; then

steps/get_train_ctm.sh $dataset_path model/lang_aishell $dataset_path/tmp/nnet3_align || exit 1

fi

exit 0

