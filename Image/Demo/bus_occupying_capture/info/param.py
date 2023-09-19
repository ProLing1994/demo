
def load_objectinfo():

    objectinfo = {}

    # car_info
    objectinfo['car_info'] = {}
    objectinfo['car_info']['roi'] = []
    objectinfo['car_info']['attri'] = 'none'
    objectinfo['car_info']['detect_score'] = 0.0

    # plate_info
    objectinfo['plate_info'] = {}
    objectinfo['plate_info']['roi'] = []
    objectinfo['plate_info']['kind_roi'] = []
    objectinfo['plate_info']['num_roi'] = []
    objectinfo['plate_info']['kind'] = ''
    objectinfo['plate_info']['num'] = ''
    objectinfo['plate_info']['column'] = 'none'
    objectinfo['plate_info']['country'] = 'none'
    objectinfo['plate_info']['city'] = 'none'
    objectinfo['plate_info']['car_type'] = 'none'
    objectinfo['plate_info']['color'] = 'none'
    objectinfo['plate_info']['score'] = 0.0
    objectinfo['plate_info']['dist_left_lane_line'] = 0.0
    objectinfo['plate_info']['dist_right_lane_line'] = 0.0
    objectinfo['plate_info']['ignore'] = False

    # face_info
    objectinfo['face_info'] = {}
    objectinfo['face_info']['roi'] = []
    objectinfo['face_info']['landmark'] = []
    objectinfo['face_info']['landmark_degree'] = 0.0
    objectinfo['face_info']['landmark_positive_cls'] = 0

    # state
    objectinfo['state'] = {}
    objectinfo['state']['stable_loc'] = []                                  # 稳定坐标
    objectinfo['state']['center_point_list'] = []                           # 车辆中心点轨迹（多帧）

    objectinfo['state']['frame_num'] = 0                                    # 进入画面帧数
    objectinfo['state']['disappear_frame_num'] = 0                          # 消失画面帧数

    objectinfo['state']['up_down_speed'] = 0                                # 车辆速度（上下行）
    objectinfo['state']['left_right_speed'] = 0                             # 车辆速度（左右行）
    objectinfo['state']['up_down_state'] = 'Stop'                           # 车辆状态（上下行）
    objectinfo['state']['up_down_state_frame_num'] = 0                      # 车辆状态（上下行）帧数
    objectinfo['state']['left_right_state'] = 'Stop'                        # 车辆状态（左右行）
    objectinfo['state']['left_right_state_frame_num'] = 0                   # 车辆状态（左右行）帧数

    objectinfo['state']['obj_num'] = 0                                      # 车牌识别帧数
    objectinfo['state']['obj_disappear_num'] = 0                            # 车牌消失帧数
    objectinfo['state']['lpr_kind_list'] = []                               # 车牌识别结果（多帧）
    objectinfo['state']['lpr_num_list'] = []                                # 车牌识别结果（多帧）
    objectinfo['state']['lpr_score_list'] = []                              # 车牌识别结果得分（多帧）
    objectinfo['state']['lpr_column_list'] = []                             # 车牌识别结果得分（多帧）
    objectinfo['state']['lpr_color_list'] = []                              # 车牌识别结果得分（多帧）
    objectinfo['state']['lpr_country_list'] = []                            # 车牌识别结果得分（多帧）
    objectinfo['state']['lpr_city_list'] = []                               # 车牌识别结果得分（多帧）
    objectinfo['state']['lpr_car_type_list'] = []                           # 车牌识别结果得分（多帧）
    objectinfo['state']['face_landmark_degree_list'] = []                   # 人脸角度识别结果得分（多帧）
    objectinfo['state']['face_landmark_positive_cls_list'] = []             # 人脸角度识别结果得分（多帧）

    # capture
    objectinfo['capture'] = {}
    objectinfo['capture']['far_report_flage'] = False                        # 抓拍标志位
    objectinfo['capture']['near_report_flage'] = False                       # 抓拍标志位
    objectinfo['capture']['left_report_flage'] = False                       # 抓拍标志位
    objectinfo['capture']['right_report_flage'] = False                      # 抓拍标志位
    objectinfo['capture']['outtime_flage_01'] = False                        # 抓拍标志位
    objectinfo['capture']['outtime_flage_02'] = False                        # 抓拍标志位
    objectinfo['capture']['outtime_flage_double_01'] = False                 # 抓拍标志位

    objectinfo['capture']['flage'] = ''                                      # 抓拍标志信息
    objectinfo['capture']['capture_frame_num'] = 0                           # 抓拍帧数
    objectinfo['capture']['img_bbox_info_list'] = []                         # 抓拍结果
    objectinfo['capture']['capture_bool'] = False                            # 抓拍成功标志
    objectinfo['capture']['draw_bool'] = False                               # 抓拍成功标志

    # state
    objectinfo['state_occupying'] = {}
    objectinfo['state_occupying']['frame_num'] = 0                           # 进入占道区域帧数
    objectinfo['state_occupying']['disappear_frame_num'] = 0                 # 消失画面帧数
    objectinfo['state_occupying']['car_line'] = 0                            
    objectinfo['state_occupying']['intersect_point'] = 0                     
    
    # capture_occupying
    objectinfo['capture_occupying'] = {}
    objectinfo['capture_occupying']['cap_lpr_num']= ''                       # 抓拍车牌识别结果
    objectinfo['capture_occupying']['cap_lpr_color']= ''                     # 抓拍车牌识别结果
    objectinfo['capture_occupying']['cap_bool']= False                       # 抓拍车牌成功标志
    objectinfo['capture_occupying']['cap_img'] = []
    objectinfo['capture_occupying']['cap_plate_roi'] = []
    objectinfo['capture_occupying']['capture_bool'] = False                  # 抓拍成功标志
    objectinfo['capture_occupying']['draw_bool'] = False                     # 抓拍成功标志

    # track_id
    objectinfo['track_id'] = -1

    return objectinfo