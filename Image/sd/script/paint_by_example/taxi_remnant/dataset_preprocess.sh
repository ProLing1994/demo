data_dir=/yuanhuan/data/image/Taxi_remnant/original/shenzhen
crop_data_dir=/yuanhuan/data/image/Taxi_remnant/training/sd_crop_clip_sam_bottle_all_0913/shenzhen
clip_ref_class=bottle
# data_dir=/yuanhuan/data/image/Taxi_remnant/original/remnant_example
# crop_data_dir=/yuanhuan/data/image/Taxi_remnant/training/remnant_example/
# clip_ref_class=all

#################################
# step 1：处理标注数据（增加 mask SAM 自动分割，训练脚本修改）
#################################
# done
# date_name_list=(20230616 20230626 20230628 20230702 20230703 20230704 BYDe6_middle_20230720 BYDe6_side_20230720 Camry_middle_20230719 Camry_middle_20230720 Camry_side_20230719 Fox_20230809 Havel_20230810 Lexus_20230810 MKZ_20230807 MKZ_middle_20230721 MKZ_side_20230721 Pickup_middle_20230721 SUV_20230809)

# todo
# date_name_list=(20230616 20230626 20230628 20230702 20230703 20230704 BYDe6_middle_20230720 BYDe6_side_20230720 Camry_middle_20230719 Camry_middle_20230720 Camry_side_20230719 Fox_20230809 Havel_20230810 Lexus_20230810 MKZ_20230807 MKZ_middle_20230721 MKZ_side_20230721 Pickup_middle_20230721 SUV_20230809)
date_name_list=(20230616 20230626 20230628 20230702 20230703 20230704 BYDe6_middle_20230720 BYDe6_side_20230720 Camry_middle_20230719 Camry_middle_20230720 Camry_side_20230719 Fox_20230809 Havel_20230810 Lexus_20230810 MKZ_20230807 MKZ_middle_20230721 MKZ_side_20230721 Pickup_middle_20230721 SUV_20230809)

for date_name in ${date_name_list[@]}; do 
    echo $date_name
    
    # json -> xml
    python /yuanhuan/code/demo/Image/Basic/script/json/platform_json_to_xml.py --input_dir=$data_dir/$date_name --jpg_name=image --json_name=json_v0 --xml_name=xml

    # 根据 taxi_remnant 位置，裁剪图像区域，指定类别 bottle，获得 参考图 reference，掩码图 mask、mask img，以及 resize 图像
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset/prepare_dataset.py --date_name=$date_name --input_dir=$data_dir --output_dir=$crop_data_dir --clip_ref_class=$clip_ref_class
    # 训练集测试集划分
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_train_test_split/data_train_test_split.py  --date_name=$date_name --input_dir=$crop_data_dir
    # 收集参考图像
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_move/data_reference.py --date_name=$date_name --input_dir=$crop_data_dir --output_dir=$crop_data_dir
done

python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_train_test_split/data_train_test_split_merge.py --input_dir=$crop_data_dir