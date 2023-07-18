data_dir=/yuanhuan/data/image/RM_C28_safeisland/original/

seg_name=safeisland_mask_202307
training_data_dir=/yuanhuan/data/image/RM_C28_safeisland/training/

#################################
# step 1：处理标注数据
#################################
# done
# date_name_list=(america zhongdong)

# todo
date_name_list=(america zhongdong)

for date_name in ${date_name_list[@]}; do 
    echo $date_name

    python /yuanhuan/code/demo/Image/segmentation2d/script/dataset/dataset_safeisland/dataset_c28/dataset_mask/gen_seg_mask.py --date_name=$date_name --input_dir=$data_dir
    python /yuanhuan/code/demo/Image/segmentation2d/script/dataset/dataset_safeisland/dataset_c28/dataset_train_test_split/data_train_test_split.py --date_name=$date_name --input_dir=$data_dir
    python /yuanhuan/code/demo/Image/segmentation2d/script/dataset/dataset_safeisland/dataset_c28/dataset_move/data_move.py --date_name=$date_name --input_dir=$data_dir --seg_name=$seg_name --output_dir=$training_data_dir
done