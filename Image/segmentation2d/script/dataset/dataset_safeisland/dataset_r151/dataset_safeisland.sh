data_dir=/yuanhuan/data/image/RM_R151_safeisland/original

#################################
# step 1：处理收集数据
#################################
# 王东兴提供数据：原始 img、分割图 mask

# done
# date_name_list=(avm_right_20230402_segonly avm_left_20230408_segonly avm_left_20230402_segonly avm_left_2d1l_230213 avm_front_20230408_segonly avm_front_20230402_segonly avm_front_2d1l_230213_p2 avm_front_2d1l_230213_p1 20220926_C53AB_WA_detseg_fix 20220821_C53AB_WA_detseg_fix 20220819_C53AB_WA_detseg 20220816_C53AB_wideangle_detseg 20220804_C53_WA_detseg 20220708_c53_WideAngle_detseg 20220707_C53AX_WA_detseg 20220706_c53_wideangle_detseg_fix 20220630_c53_wideangle_detseg 20220628_c53_WideAngle_detseg 20220620_C53AX_WA_detseg)

# todo
date_name_list=()

for date_name in ${date_name_list[@]}; do 
    echo $date_name

    python /yuanhuan/code/demo/Image/segmentation2d/script/dataset/dataset_safeisland/dataset_r151/dataset_check/dataset_check.py --date_name=$date_name --input_dir=$data_dir
    python /yuanhuan/code/demo/Image/segmentation2d/script/dataset/dataset_safeisland/dataset_r151/dataset_mask/gen_seg_mask.py --date_name=$date_name --input_dir=$data_dir
    python /yuanhuan/code/demo/Image/segmentation2d/script/dataset/dataset_safeisland/dataset_r151/dataset_train_test_split/data_train_test_split.py --date_name=$date_name --input_dir=$data_dir

done