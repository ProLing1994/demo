python demo.py --video_dir=".\\test\\avi" --output_video_dir=".\\test\\avi_video_capture" --suffix=".avi" --steps="1,3" --model_path=".\\model\\ssd_rfb\\SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-07-22-00.xml"

可选参数：
--video_dir 输入视频地址，ex：--video_dir=".\\test\\avi"
--output_video_dir 输出视频地址，ex：--output_video_dir=".\\test\\avi_video_capture"
--suffix 视频后缀，支持 mp4、avi，ex：--suffix=".avi"
--steps 步骤共 4 步，1：车辆抓取，2：视频合并并剪裁，3：挑选黄色车牌或车型为货车和巴士车辆并剪裁，ex：--steps="1,2"
--model_path 模型路径，ex：--model_path=".\\model\\ssd_rfb\\SSD_VGG_FPN_RFB_VOC_car_bus_truck_licenseplate_softmax_zg_2022-07-22-00.xml"
--GPU 是否启用 GPU 模式（处理速度会更快），ex：--GPU

常用裁剪方式：
淮北高速卡口1、卡口2、卡口3：--steps=1,3，理由：尽可能挑选多的数据，挑选黄色车牌或车型为货车和巴士车辆并剪裁
桐乡三道防线：--steps=1,3，理由：尽可能挑选多的数据，挑选黄色车牌或车型为货车和巴士车辆并剪裁