#!/bin/bash

data_dir=/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil/
data_csv_dir=/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_csv/
data_crop_dir=/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_crop/
error_data_dir=/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_error_data/
analysis_dir=/yuanhuan/data/image/RM_ANPR/original/Brazil/Brazil_analysis/
data_diffste_dir=/yuanhuan/data/image/RM_ANPR/original/Brazil/DIFFSTE/

ocr_name=plate_brazil_202308
ocr_diffste_name=$ocr_name
training_data_dir=/yuanhuan/data/image/RM_ANPR/training/


#################################
# step 1：处理标注数据
#################################

# done
# date_name_list=(Brazil_new_style)

# todo
date_name_list=(Brazil_new_style)

for date_name in ${date_name_list[@]}; do 
    echo $date_name

    # normal
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_brazil/dataset_check/dataset_csv.py --date_name=$date_name --input_dir=$data_dir --output_csv_dir=$data_csv_dir --output_crop_data_dir=$data_crop_dir --output_error_data_dir=$error_data_dir --bool_write_error_data --bool_write_crop_data
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_brazil/dataset_analysis/analysis_dataset_label_num.py --date_name=$date_name --input_csv_dir=$data_csv_dir --output_analysis_dir=$analysis_dir

done

python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_brazil/dataset_analysis/analysis_dataset_label_num_merge.py --input_dir=$analysis_dir


#################################
# step 2：生成训练数据 ocr
#################################

# done
# date_name_list=(Brazil_new_style)
# date_name_list=(diffste_963_old_new_style original_400_old_style original_563_new_style)

# todo
date_name_list=(Brazil_new_style)

for date_name in ${date_name_list[@]}; do 
    echo $date_name
    
#     # ocr
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_brazil/dataset_mask/gen_ocr_img.py --date_name=$date_name --ocr_name=$ocr_name --input_csv_dir=$data_csv_dir --output_dir=$training_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_brazil/dataset_train_test_split/data_train_test_split_ocr.py --date_name=$date_name --ocr_name=$ocr_name --input_dir=$training_data_dir

    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_brazil/dataset_mask/gen_ocr_img_augment.py --date_name=$date_name --ocr_name=$ocr_name --output_dir=$training_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_brazil/dataset_train_test_split/data_train_test_split_ocr_augment.py --date_name=$date_name --ocr_name=$ocr_name --input_dir=$training_data_dir

    # # diffste ocr
    # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_brazil/dataset_mask/gen_ocr_img_diffste.py --date_name=$date_name --ocr_name=$ocr_diffste_name --input_dir=$data_diffste_dir --output_dir=$training_data_dir
    # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_brazil/dataset_train_test_split/data_train_test_split_ocr.py --date_name=$date_name --ocr_name=$ocr_diffste_name --input_dir=$training_data_dir

done

# ocr
python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_brazil/dataset_train_test_split/data_train_test_split_ocr_merge.py --ocr_name=$ocr_name --input_dir=$training_data_dir

# label split 标签划分，区分单双行车牌
python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_brazil/dataset_label_split/data_label_split.py --ocr_name=$ocr_name --input_dir=$training_data_dir


#################################
# step 3：2 paddleocr label
#################################
# 2 paddleocr label
# paddle_ocr_name=$ocr_name
paddle_ocr_name=$ocr_diffste_name
image_set_name=ImageSetsOcrLabel_single_line
paddle_ocr_data_dir=/yuanhuan/model/image/lpr/paddle_dict/$paddle_ocr_name
python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/lpr_to_paddleocr_label.py --input_dir=$training_data_dir/$paddle_ocr_name --image_set_name=$image_set_name --output_dir=$paddle_ocr_data_dir
python /yuanhuan/code/demo/Image/recognition2d/script/paddle/dataset/label_dict.py --output_dir=$paddle_ocr_data_dir --output_name=brazil_dict.txt --data_dict_name=script.lpr.dataset.dataset_brazil.dataset_dict.dataset_brazil_dict_normal