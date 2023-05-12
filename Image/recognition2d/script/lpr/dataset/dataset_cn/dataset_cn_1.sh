data_dir=/yuanhuan/data/image/RM_ANPR/original/cn/china/
data_csv_dir=/yuanhuan/data/image/RM_ANPR/original/cn/china_csv/
data_crop_dir=/yuanhuan/data/image/RM_ANPR/original/cn/china_crop/
error_data_dir=/yuanhuan/data/image/RM_ANPR/original/cn/china_error_data/
analysis_dir=/yuanhuan/data/image/RM_ANPR/original/cn/china_analysis/

seg_name=seg_cn_202305
ocr_name=plate_cn_202305
training_data_dir=/yuanhuan/data/image/RM_ANPR/training/


# #################################
# # step 1：处理标注数据
# #################################

# # done
# # date_name_list=(jilin liaoning neimenggu ningxia qinghai shandong shanghai shanxi shanxi_jin sichuan tianjin xianggangaomen xinjiang yunnan zhejiang yellow_license_plate 特殊车牌 ZG)

# # todo
# date_name_list=()

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # normal
#     # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_check/dataset_csv.py --date_name=$date_name --input_dir=$data_dir --output_csv_dir=$data_csv_dir --output_crop_data_dir=$data_crop_dir --output_error_data_dir=$error_data_dir --bool_write_error_data
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_check/dataset_csv.py --date_name=$date_name --input_dir=$data_dir --output_csv_dir=$data_csv_dir --output_crop_data_dir=$data_crop_dir --output_error_data_dir=$error_data_dir --bool_write_error_data --bool_check_img
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_analysis/analysis_dataset_label_num.py --date_name=$date_name --input_csv_dir=$data_csv_dir --output_analysis_dir=$analysis_dir

# done

# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_analysis/analysis_dataset_label_num_merge.py --input_dir=$analysis_dir

# #################################
# # step 2：生成训练数据 seg
# #################################

# # done
# # date_name_list=(jilin liaoning neimenggu ningxia qinghai shandong shanghai shanxi shanxi_jin sichuan tianjin xianggangaomen xinjiang yunnan zhejiang yellow_license_plate 特殊车牌 ZG)

# # todo
# date_name_list=(jilin liaoning neimenggu ningxia qinghai shandong shanghai shanxi shanxi_jin sichuan tianjin)

# for date_name in ${date_name_list[@]}; do 
#     echo $date_name

#     # # seg
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_mask/gen_seg_mask.py --date_name=$date_name --seg_name=$seg_name --input_csv_dir=$data_csv_dir --output_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_train_test_split/data_train_test_split_seg.py --date_name=$date_name --seg_name=$seg_name --input_dir=$training_data_dir

#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_mask/gen_seg_mask_augment.py --date_name=$date_name --seg_name=$seg_name --output_dir=$training_data_dir
#     python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_train_test_split/data_train_test_split_seg_augment.py --date_name=$date_name --seg_name=$seg_name --input_dir=$training_data_dir

# done

# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_train_test_split/data_train_test_split_seg_merge.py --seg_name=$seg_name --input_dir=$training_data_dir


#################################
# step 3：生成训练数据 ocr
#################################

# done
# date_name_list=(jilin liaoning neimenggu ningxia qinghai shandong shanghai shanxi shanxi_jin sichuan tianjin xianggangaomen xinjiang yunnan zhejiang yellow_license_plate 特殊车牌 ZG)

# todo
date_name_list=(jilin liaoning neimenggu ningxia qinghai shandong shanghai shanxi shanxi_jin sichuan tianjin)

for date_name in ${date_name_list[@]}; do 
    echo $date_name
    
    # ocr
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_mask/gen_ocr_img.py --date_name=$date_name --ocr_name=$ocr_name --input_csv_dir=$data_csv_dir --output_dir=$training_data_dir
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_train_test_split/data_train_test_split_ocr.py --date_name=$date_name --ocr_name=$ocr_name --input_dir=$training_data_dir

    # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_mask/gen_ocr_img_augment.py --date_name=$date_name --ocr_name=$ocr_name --output_dir=$training_data_dir
    # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_train_test_split/data_train_test_split_ocr_augment.py --date_name=$date_name --ocr_name=$ocr_name --input_dir=$training_data_dir
    
done

# python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_cn/dataset_train_test_split/data_train_test_split_ocr_merge.py --ocr_name=$ocr_name --input_dir=$training_data_dir