python demo.py --video_dir=".\\test\\avi" --output_video_dir=".\\test\\avi_video_capture" --suffix=".avi" --steps="1,2" --model_path=".\\model\\ssd_mobilenetv2_fpn.xml"

可选参数：
--video_dir 输入视频地址，ex：--video_dir=".\\test\\avi"
--output_video_dir 输出视频地址，ex：--output_video_dir=".\\test\\avi_video_capture"
--suffix 视频后缀，支持 mp4、avi，ex：--suffix=".avi"
--steps 步骤共 2 步，1：车辆抓取，2：视频合并并剪裁，ex：--steps="1,2"
--model_path 模型路径，ex：--model_path=".\\model\\ssd_mobilenetv2_fpn.xml"
--GPU 是否启用 GPU 模式（处理速度会更快），ex：--GPU