#!/bin/bash

data_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_old_style/
# data_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_new_style/
data_csv_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_csv/
data_crop_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop/
data_crop_csv_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop_csv/
error_data_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_error_data/
error_crop_data_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_error_crop_data/
analysis_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_analysis/

seg_name=seg_zd_202306
ocr_name=plate_zd_mask_202306
training_data_dir=/yuanhuan/data/image/RM_ANPR/training/


#################################
# step 1：处理标注数据
#################################

# done
# date_name_list=(uae_20220804_0809 uae_20220810_0811 uae_20220828_0831 uae_20220901_0903 uae_20220904_0905 uae_20221024_1029 uae_20221024_1029_1080p uae_20221115_1116 uae_20221115_1116_1080p uae_2022_0 uae_2022_1 uae_2022_2 uae_2022_3 uae_2022_4 uae_2022_5 uae_2022_6 uae_2022_7 uae_2022_8 uae_2022_city_AJMAN_0 uae_2022_city_FUJAIRAH_0 uae_2022_city_RAK_0 uae_2022_city_SHARJAH_0 uae_2022_city_UMMALQAIWAIN_0 uae_2022_color_green_0 uae_2022_color_yellow_0)
# date_name_list=(shate_20230308 shate_20230309)

# todo
date_name_list=(uae_2022_old)

for date_name in ${date_name_list[@]}; do 
    echo $date_name

    # normal
    # old style 
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_check/dataset_csv.py --date_name=$date_name --input_dir=$data_dir --output_csv_dir=$data_csv_dir --output_crop_data_dir=$data_crop_dir --output_error_data_dir=$error_data_dir --bool_write_error_data --bool_write_crop_data --bool_check_img
    # new style
    # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_check/dataset_csv.py --date_name=$date_name --input_dir=$data_dir --output_csv_dir=$data_csv_dir --output_crop_data_dir=$data_crop_dir --output_error_data_dir=$error_data_dir --new_style --bool_write_error_data --bool_write_crop_data --bool_check_img

    # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_analysis/analysis_dataset_label_num.py --date_name=$date_name --input_csv_dir=$data_csv_dir --output_analysis_dir=$analysis_dir

done

# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_analysis/analysis_dataset_label_num_merge.py --input_dir=$analysis_dir


# #################################
# # step 2：生成预标注标签 seg
# #################################

# # done
# # date_name_list=()

# # todo
# date_name_list=(shate_20230308)

# json_name=Jsons_Prelabel
# xml_name=Xmls_Prelabel

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # seg
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_pre_label/dataset_pre_label_seg_mask.py --date_name=$date_name --input_crop_data_dir=$data_crop_dir --json_name=$json_name
#     python /yuanhuan/code/demo/Image/Basic/script/json/platform_json_to_xml.py --input_dir=$data_crop_dir/$date_name --jpg_name=Images --json_name=$json_name --xml_name=$xml_name
# done


# #################################
# # step 3：生成训练数据 seg
# #################################

# # done
# # date_name_list=(uae_20220804_0809 uae_20220810_0811 uae_20220828_0831 uae_20220901_0903 uae_20220904_0905 uae_20221024_1029 uae_20221024_1029_1080p uae_20221115_1116 uae_20221115_1116_1080p uae_2022_0 uae_2022_1 uae_2022_2 uae_2022_3 uae_2022_4 uae_2022_5 uae_2022_6 uae_2022_7 uae_2022_8 uae_2022_city_AJMAN_0 uae_2022_city_FUJAIRAH_0 uae_2022_city_RAK_0 uae_2022_city_SHARJAH_0 uae_2022_city_UMMALQAIWAIN_0 uae_2022_color_green_0 uae_2022_color_yellow_0)
# # date_name_list=(shate_20230308 shate_20230309)

# # todo
# date_name_list=(shate_20230308)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # old style 
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_check/dataset_crop_csv.py --date_name=$date_name --input_csv_dir=$data_csv_dir --input_crop_data_dir=$data_crop_dir --output_csv_dir=$data_crop_csv_dir --output_error_crop_data_dir=$error_crop_data_dir --bool_write_error_data
#     # new style
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_check/dataset_crop_csv.py --date_name=$date_name --input_csv_dir=$data_csv_dir --input_crop_data_dir=$data_crop_dir --output_csv_dir=$data_crop_csv_dir --output_error_crop_data_dir=$error_crop_data_dir --new_style --bool_write_error_data
    
#     # seg
#     # old style 
#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_mask/gen_seg_mask.py --date_name=$date_name --seg_name=$seg_name --input_csv_dir=$data_crop_csv_dir --output_dir=$training_data_dir
#     # new style
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_mask/gen_seg_mask.py --date_name=$date_name --seg_name=$seg_name --input_csv_dir=$data_crop_csv_dir --output_dir=$training_data_dir --new_style

#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_seg.py --date_name=$date_name --seg_name=$seg_name --input_dir=$training_data_dir

#     # old style 
#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_mask/gen_seg_mask_augment.py --date_name=$date_name --seg_name=$seg_name --output_dir=$training_data_dir
#     # new style
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_mask/gen_seg_mask_augment.py --date_name=$date_name --seg_name=$seg_name --output_dir=$training_data_dir --new_style

#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_seg_augment.py --date_name=$date_name --seg_name=$seg_name --input_dir=$training_data_dir

# done

# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_seg_merge.py --seg_name=$seg_name --input_dir=$training_data_dir


# ################################
# # step 4：生成训练数据 ocr
# ################################

# # done
# # date_name_list=(uae_20220804_0809 uae_20220810_0811 uae_20220828_0831 uae_20220901_0903 uae_20220904_0905 uae_20221024_1029 uae_20221024_1029_1080p uae_20221115_1116 uae_20221115_1116_1080p uae_2022_0 uae_2022_1 uae_2022_2 uae_2022_3 uae_2022_4 uae_2022_5 uae_2022_6 uae_2022_7 uae_2022_8 uae_2022_city_AJMAN_0 uae_2022_city_FUJAIRAH_0 uae_2022_city_RAK_0 uae_2022_city_SHARJAH_0 uae_2022_city_UMMALQAIWAIN_0 uae_2022_color_green_0 uae_2022_color_yellow_0)
# # date_name_list=(shate_20230308 shate_20230309)

# # todo
# date_name_list=(shate_20230308)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name
    
#     # ocr
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_mask/gen_ocr_img.py --date_name=$date_name --seg_name=$seg_name --ocr_name=$ocr_name --output_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_ocr.py --date_name=$date_name --ocr_name=$ocr_name --input_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_mask/gen_ocr_img_augment.py --date_name=$date_name --seg_name=$seg_name --ocr_name=$ocr_name --output_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_ocr_augment --date_name=$date_name --ocr_name=$ocr_name --input_dir=$training_data_dir
    
# done

# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_train_test_split/data_train_test_split_ocr_merge.py --ocr_name=$ocr_name --input_dir=$training_data_dir