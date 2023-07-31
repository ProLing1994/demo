data_dir=/yuanhuan/data/image/RM_upspeed/original/
crop_data_dir=/yuanhuan/data/image/RM_upspeed/crop/
train_data_dir=/yuanhuan/data/image/RM_upspeed/training/

#################################
# step 1：处理标注数据
#################################
# done
# date_name_list=(Europe China America_500w America_200w)

# todo
date_name_list=(America_200w)

for date_name in ${date_name_list[@]}; do 
    echo $date_name

    # 根据 upspeed 位置，裁剪图像区域
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/upspeed/dataset/crop_image_xml.py --date_name=$date_name --input_dir=$data_dir --output_dir=$crop_data_dir
    # 生成重绘用到的 mask 和 refeneren 图像
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/upspeed/dataset/gen_inpaint_mask.py --date_name=$date_name --input_dir=$crop_data_dir
    # 生成 Fintune 模型需要用到的 bbox
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/upspeed/dataset/read_bbox.py --date_name=$date_name --input_dir=$crop_data_dir --output_dir=$train_data_dir
    # 训练集测试集划分
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/upspeed/dataset_train_test_split/data_train_test_split.py  --date_name=$date_name --input_dir=$train_data_dir
    # 将数据整理成训练用到的格式
    python /yuanhuan/code/demo/Image/sd/script/paint_by_example/upspeed/dataset_move/data_move.py --date_name=$date_name --input_dir=$train_data_dir --output_dir=$train_data_dir

done