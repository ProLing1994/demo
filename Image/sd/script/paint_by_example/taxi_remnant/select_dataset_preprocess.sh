find_dir=/yuanhuan/data/image/Taxi_remnant/original/shenzhen

data_dir=/yuanhuan/data/image/Taxi_remnant/original_select
crop_data_dir=/yuanhuan/data/image/Taxi_remnant/training/sd_crop_sd_sam_0915/
# crop_data_dir=/yuanhuan/data/image/Taxi_remnant/training/sd_crop_select_sam_0828/
# data_dir=/yuanhuan/data/image/Taxi_remnant/original_sd
clip_ref_class=all

#################################
# step 1：处理标注数据（增加 mask SAM 自动分割，训练脚本修改）
#################################
# done
# date_name_list=(bottle umbrella wallet book handbag)

# todo
date_name_list=(wallet)

for date_name in ${date_name_list[@]}; do 
    echo $date_name
    
    # # find -> xml
    # python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_move/data_xml_find.py --date_name=$date_name --input_dir=$data_dir --find_dir=$find_dir

    # 根据 taxi_remnant 位置，裁剪图像区域，指定类别 bottle，获得 参考图 reference，掩码图 mask、mask img，以及 resize 图像
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset/prepare_dataset.py --date_name=$date_name --input_dir=$data_dir --output_dir=$crop_data_dir --clip_ref_class=$clip_ref_class
    # 训练集测试集划分
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_train_test_split/data_train_test_split.py  --date_name=$date_name --input_dir=$crop_data_dir
done