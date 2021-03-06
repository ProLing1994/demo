#!/bin/bash

# 该脚本进行模型测试

stage=1

# # init
# # xiaorui
# config_file=/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_13_res15_narrow_fintune_kd_12162020/kws_config_xiaorui.py
# csv_path=/mnt/huanyuan/data/speech/Real_vehicle_sample/20201218/Real_vehicle_sample_20201218.csv
# echo "test.sh"

# # test model normal_driving & False
# if [ $stage -le 1 ];then
#     python ../../infer/test_streaming_wav.py --mode "1" --csv_path $csv_path --type "normal_driving" --config_file $config_file || exit 1
#     python ./static_result.py --config_file $config_file --csv_path $csv_path --type "normal_driving" || exit 1
# fi

# # test model normal_driving & True
# if [ $stage -le 2 ];then
#     python ../../infer/test_streaming_wav.py --mode "1" --csv_path $csv_path --type "normal_driving" --bool_noise_reduction --config_file $config_file || exit 1
#     python ./static_result.py --config_file $config_file --csv_path $csv_path --type "normal_driving" --bool_noise_reduction || exit 1
# fi

# # test model idling_driving & False
# if [ $stage -le 3 ];then
#     python ../../infer/test_streaming_wav.py --mode "1" --csv_path $csv_path --type "idling_driving" --config_file $config_file || exit 1
#     python ./static_result.py --config_file $config_file --csv_path $csv_path --type "idling_driving" || exit 1
# fi

# # test model idling_driving & True
# if [ $stage -le 4 ];then
#     python ../../infer/test_streaming_wav.py --mode "1" --csv_path $csv_path --type "idling_driving" --bool_noise_reduction --config_file $config_file || exit 1
#     python ./static_result.py --config_file $config_file --csv_path $csv_path --type "idling_driving" --bool_noise_reduction || exit 1
# fi

# init
# xiaorui
config_file=/mnt/huanyuan/model/model_10_30_25_21/model/kws_xiaorui1_10_res15_finetune_12162020/kws_config_xiaorui.py
csv_path=/mnt/huanyuan/data/speech/Recording_sample/Real_vehicle_sample/20210105/Real_vehicle_sample_20210105.csv
echo "test.sh"

# test model 降噪前 & False
if [ $stage -le 1 ];then
    python ../../infer/test_streaming_wav.py --mode "1" --csv_path $csv_path --type "降噪前" --config_file $config_file || exit 1
    python ./static_result.py --config_file $config_file --csv_path $csv_path --type "降噪前" || exit 1
fi

# test model 降噪后-SPD & True
if [ $stage -le 2 ];then
    python ../../infer/test_streaming_wav.py --mode "1" --csv_path $csv_path --type "降噪后-SPD" --bool_noise_reduction --config_file $config_file || exit 1
    python ./static_result.py --config_file $config_file --csv_path $csv_path --type "降噪后-SPD" --bool_noise_reduction || exit 1
fi

# test model 降噪后-GSC & True
if [ $stage -le 3 ];then
    python ../../infer/test_streaming_wav.py --mode "1" --csv_path $csv_path --type "降噪后-GSC" --bool_noise_reduction --config_file $config_file || exit 1
    python ./static_result.py --config_file $config_file --csv_path $csv_path --type "降噪后-GSC" --bool_noise_reduction || exit 1
fi

echo "test.sh succeeded"