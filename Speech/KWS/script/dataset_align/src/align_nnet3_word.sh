#!/bin/bash

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

data=$1
nj=3
stage=0
x=kaldi_type

if [ $stage -le 0 ]; then

utils/fix_data_dir.sh $data/$x || exit 1;

steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj $data/$x $data/$x/tmp/log/make_mfcc/ $data/$x/tmp/mfcc_hires || exit 1

utils/fix_data_dir.sh $data/$x || exit 1

steps/compute_cmvn_stats.sh $data/${x} $data/$x/tmp/log/make_mfcc $data/$x/tmp/cmvn || exit 1

utils/fix_data_dir.sh $data/$x || exit 1;


fi

if [ $stage -le 1 ]; then

steps/nnet3/align.sh --nj $nj --cmd "$train_cmd" --use-gpu false --scale_opts "--transition-scale=1.0 --acoustic-scale=10.0 --self-loop-scale=0.1" \
                  $data/${x} model/lang_aishell model/nnet3/tdnn_sp/ $data/$x/tmp/nnet3_align || exit 1

fi


if [ $stage -le 2 ]; then

steps/get_train_ctm.sh $data/$x model/lang_aishell $data/$x/tmp/nnet3_align || exit 1

fi

exit 0

