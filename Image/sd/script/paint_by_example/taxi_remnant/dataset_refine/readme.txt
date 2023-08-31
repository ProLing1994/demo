# 利用 finetune 模型以及 sam 分割物体边界，按照参考图重绘
/yuanhuan/code/demo/Image/sd/paint_by_example/scripts/test/inference_w_crop_sam.py

# 参考图重绘后，存在三个问题：1、非同类型物体替换；2、sam 分割物体失败；3、替换效果不好
# 人工对重绘后图像 Grids 进行挑选，生成 Grids_select 文件夹，整理文件，生成最终检测用的数据
/yuanhuan/code/demo/Image/sd/script/paint_by_example/taxi_remnant/dataset_refine/dataset_refine.py