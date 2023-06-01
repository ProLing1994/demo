data_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE/
data_csv_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_csv/
data_crop_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_crop_new/
error_data_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_error_data/
analysis_dir=/yuanhuan/data/image/RM_ANPR/original/zd/UAE/UAE_analysis/

seg_name=seg_zd_202305
ocr_name=plate_zd_202305
training_data_dir=/yuanhuan/data/image/RM_ANPR/training/


#################################
# step 1：处理标注数据
#################################

# done
# date_name_list=(shate_20230308)

# todo
date_name_list=(shate_20230308)

for date_name in ${date_name_list[@]}; do 
    echo $date_name

    # normal
    # python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_check/dataset_csv.py --date_name=$date_name --input_dir=$data_dir --output_csv_dir=$data_csv_dir --output_crop_data_dir=$data_crop_dir --output_error_data_dir=$error_data_dir --bool_write_error_data --bool_write_crop_data
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_check/dataset_csv.py --date_name=$date_name --input_dir=$data_dir --output_csv_dir=$data_csv_dir --output_crop_data_dir=$data_crop_dir --output_error_data_dir=$error_data_dir --bool_write_error_data --bool_write_crop_data --bool_check_img
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_analysis/analysis_dataset_label_num.py --date_name=$date_name --input_csv_dir=$data_csv_dir --output_analysis_dir=$analysis_dir

done

python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_analysis/analysis_dataset_label_num_merge.py --input_dir=$analysis_dir


#################################
# step 2：生成预标注标签 seg
#################################

# done
# date_name_list=()

# todo
date_name_list=(shate_20230308)

json_name=Jsons_Prelabel
xml_name=Xmls_Prelabel

for date_name in ${date_name_list[@]}; do 
    echo $date_name

    # seg
    python /yuanhuan/code/demo/Image/recognition2d/script/lpr/dataset/dataset_zd/dataset_pre_label/dataset_pre_label_seg_mask.py --date_name=$date_name --input_crop_data_dir=$data_crop_dir --json_name=$json_name
    python /yuanhuan/code/demo/Image/Basic/script/json/platform_json_to_xml.py --input_dir=$data_crop_dir/$date_name --jpg_name=Images --json_name=$json_name --xml_name=$xml_name
done