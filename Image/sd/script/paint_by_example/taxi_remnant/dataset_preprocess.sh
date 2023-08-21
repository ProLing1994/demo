data_dir=/yuanhuan/data/image/Taxi_remnant/original/shenzhen
# crop_data_dir=/yuanhuan/data/image/Taxi_remnant/sd_crop_0810/shenzhen
# train_data_dir=/yuanhuan/data/image/Taxi_remnant/sd_training_0810/shenzhen
crop_data_dir=/yuanhuan/data/image/Taxi_remnant/sd_crop_bottle_all_0815/shenzhen
train_data_dir=/yuanhuan/data/image/Taxi_remnant/sd_training_bottle_all_0815/shenzhen

#################################
# step 1：处理标注数据
#################################
# done
# date_name_list=(20230616 20230626 20230628 20230702 20230703 20230704 BYDe6_middle_20230720 BYDe6_side_20230720 Camry_middle_20230719 Camry_middle_20230720 Camry_side_20230719 MKZ_middle_20230721 MKZ_side_20230721 Pickup_middle_20230721)

# todo
date_name_list=(20230616 20230626 20230628 20230702 20230703 20230704 BYDe6_middle_20230720 BYDe6_side_20230720 Camry_middle_20230719 Camry_middle_20230720 Camry_side_20230719 MKZ_middle_20230721 MKZ_side_20230721 Pickup_middle_20230721)

for date_name in ${date_name_list[@]}; do 
    echo $date_name

    # # 根据 taxi_remnant 位置，裁剪图像区域（所有数据）
    # python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset/crop_image_xml.py --date_name=$date_name --input_dir=$data_dir --output_dir=$crop_data_dir
    # # 生成重绘用到的 mask 和 refeneren 图像
    # python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset/gen_inpaint_mask.py --date_name=$date_name --input_dir=$crop_data_dir
    # # 生成 Fintune 模型需要用到的 bbox
    # python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset/read_bbox.py --date_name=$date_name --input_dir=$crop_data_dir --output_dir=$train_data_dir
    # # 训练集测试集划分
    # python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_train_test_split/data_train_test_split.py  --date_name=$date_name --input_dir=$train_data_dir
    # # 将数据整理成训练用到的格式
    # python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_move/data_move.py --date_name=$date_name --input_dir=$train_data_dir --output_dir=$train_data_dir
    # # 收集参考图像
    # python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_move/data_reference.py --date_name=$date_name --input_dir=$crop_data_dir --output_dir=$crop_data_dir

    # 根据 taxi_remnant 位置，裁剪图像区域（bottle）
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset/crop_image_xml_clip.py --date_name=$date_name --input_dir=$data_dir --output_dir=$crop_data_dir
    # 生成重绘用到的 mask 和 refeneren 图像
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset/gen_inpaint_mask.py --date_name=$date_name --input_dir=$crop_data_dir --ref_name=bottle
    # 生成 Fintune 模型需要用到的 bbox
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset/read_bbox.py --date_name=$date_name --input_dir=$crop_data_dir --output_dir=$train_data_dir
    # 训练集测试集划分
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_train_test_split/data_train_test_split.py  --date_name=$date_name --input_dir=$train_data_dir
    # 将数据整理成训练用到的格式
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_move/data_move.py --date_name=$date_name --input_dir=$train_data_dir --output_dir=$train_data_dir
    # 收集参考图像
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_move/data_reference.py --date_name=$date_name --input_dir=$crop_data_dir --output_dir=$crop_data_dir
done