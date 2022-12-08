import argparse
from tkinter.tix import Tree
import cv2
import numpy as np
import sys 
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo')
from Image.Basic.utils.folder_tools import *
from Image.Demo.face_capture.demo.RMAI_FACE_API import *
from Image.Demo.face_capture.demo.RMAI_P3D_API import *
from Image.Demo.face_capture.demo.RMAI_Match_API import *
from Image.Demo.face_capture.utils.draw_tools import *
from Image.Demo.face_capture.utils.xml_tools import *


def inference_video(args):
    # mkdir 
    create_folder(args.output_video_dir)

    # face capture api
    face_capture_api = FaceCaptureApi()

    # p3d capture api
    p3d_capture_api = P3DCaptureApi()

    # match api 
    match_api = MatchApi()

    # video init 
    video_list = np.array(os.listdir(args.video_dir))
    face_video_list = video_list[[video.endswith(args.face_video_endsname) for video in video_list]]
    face_video_list.sort()
    p3d_video_list = video_list[[video.endswith(args.p3d_video_endsname) for video in video_list]]
    p3d_video_list.sort()
    assert len(face_video_list) == len(p3d_video_list)

    for idx in tqdm(range(len(face_video_list))):

        face_video_name = face_video_list[idx]
        face_video_path = os.path.join(args.video_dir, face_video_name)
        face_cap = cv2.VideoCapture(face_video_path) 
        print(int(face_cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(face_cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(face_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(face_cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧

        face_xml_name = str(face_video_name).replace(args.suffix, args.face_xml_endsname)
        face_xml_path = os.path.join(args.video_dir, face_xml_name)
        face_xml_api = XmlApi(face_xml_path, int(face_cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        p3d_video_name = p3d_video_list[idx]
        p3d_video_path = os.path.join(args.video_dir, p3d_video_name)
        p3d_cap = cv2.VideoCapture(p3d_video_path) 
        print(int(p3d_cap.get(cv2.CAP_PROP_FPS)))              # 得到视频的帧率
        print(p3d_cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 得到视频的宽
        print(p3d_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # 得到视频的高
        print(p3d_cap.get(cv2.CAP_PROP_FRAME_COUNT))           # 得到视频的总帧
        
        p3d_xml_name = str(p3d_video_name).replace(args.suffix, args.p3d_xml_endsname)
        p3d_xml_path = os.path.join(args.video_dir, p3d_xml_name)
        p3d_xml_api = XmlApi(p3d_xml_path, int(p3d_cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        assert str(face_video_name).replace(args.face_video_endsname, args.p3d_video_endsname) == p3d_video_name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        create_folder(os.path.dirname(args.output_video_dir))
        output_face_video_path = os.path.join(args.output_video_dir, face_video_list[idx])
        output_face_video = cv2.VideoWriter(output_face_video_path, fourcc, 20.0, (1920, 1080), True)
        output_p3d_video_path = os.path.join(args.output_video_dir, p3d_video_list[idx])
        output_p3d_video = cv2.VideoWriter(output_p3d_video_path, fourcc, 20.0, (1920, 1080), True)

        frame_idx = 0

        face_end_bool = False
        p3d_end_bool = False
        while True:
            
            print("\r{}/{}".format(frame_idx, max(int(face_cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(p3d_cap.get(cv2.CAP_PROP_FRAME_COUNT)) )))
            # read image
            ret, img_face = face_cap.read()
            if not ret: # if the camera over return false
                face_end_bool = True

            ret, img_p3d = p3d_cap.read()
            if not ret: # if the camera over return false
                p3d_end_bool = True

            if face_end_bool and p3d_end_bool:
                break
            
            # face capture api
            if not face_end_bool:
                face_pts = face_xml_api.find_pts(frame_idx)
                face_tracker_bboxes, face_bbox_info_list, face_bbox_state_map, face_capture_dict = face_capture_api.run(img_face, face_pts)
            else:
                face_capture_dict = {}
            
            # p3d capture api
            if not p3d_end_bool:
                p3d_pts = p3d_xml_api.find_pts(frame_idx)
                p3d_tracker_bboxes, p3d_bbox_info_list, p3d_bbox_state_map, p3d_capture_dict, p3d_roi_area = p3d_capture_api.run(img_p3d, p3d_pts)
            else:
                p3d_capture_dict = {}

            match_dict = match_api.run(face_capture_dict, p3d_capture_dict)

            if args.write_result_per_frame_bool:
                # face
                if not face_end_bool:
                    # img_face = draw_bbox_tracker(img_face, face_tracker_bboxes)
                    img_face = draw_bbox_info(img_face, face_bbox_info_list)
                    img_face = draw_bbox_state(img_face, face_bbox_state_map)
                    # p3d -> face
                    img_face = draw_bbox_state(img_face, p3d_bbox_state_map, type="p3d")
                    
                    output_face_video.write(img_face)

                # p3d
                if not p3d_end_bool:
                    # img_p3d = draw_bbox_tracker(img_p3d, p3d_tracker_bboxes)
                    img_p3d = draw_bbox_info(img_p3d, p3d_bbox_info_list, type="p3d")
                    img_p3d = draw_bbox_state(img_p3d, p3d_bbox_state_map, type="p3d")
                    img_p3d = draw_roi(img_p3d, p3d_roi_area)
                
                    output_p3d_video.write(img_p3d)

                # face
                if not face_end_bool:
                    output_img_path = os.path.join(args.output_video_dir, face_video_list[idx].replace(args.suffix, ''), face_video_list[idx].replace(args.suffix, '_{}.jpg'.format(frame_idx)))
                    create_folder(os.path.dirname(output_img_path))
                    cv2.imwrite(output_img_path, img_face)

                # p3d
                if not p3d_end_bool:
                    output_img_path = os.path.join(args.output_video_dir, p3d_video_list[idx].replace(args.suffix, ''), p3d_video_list[idx].replace(args.suffix, '_{}.jpg'.format(frame_idx)))
                    create_folder(os.path.dirname(output_img_path))
                    cv2.imwrite(output_img_path, img_p3d)

                # 保存抓拍结果
                output_capture_folder = os.path.join(args.output_video_dir, 'capture', face_video_list[idx].replace(args.suffix, ''))
                create_folder(output_capture_folder)
                draw_match_dict(match_dict, output_capture_folder)

                frame_idx += 1
        
        # vedio end
        _, _, _, face_capture_dict = face_capture_api.end_video( )
        _, _, _, p3d_capture_dict, _ = p3d_capture_api.end_video( )
        match_dict = match_api.run(face_capture_dict, p3d_capture_dict, True)

        # 保存抓拍结果
        output_capture_folder = os.path.join(args.output_video_dir, 'capture', face_video_list[idx].replace(args.suffix, ''))
        create_folder(output_capture_folder)
        draw_match_dict(match_dict, output_capture_folder)


def main():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_Face/face_test_video/原始视频/test_video/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_Face/face_test_video/原始视频/shenzhen_video_0610_1920_1080/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_Face/face_test_video/原始视频/shenzhen_video_0625_1920_1080/"
    # args.video_dir = "/mnt/huanyuan2/data/image/ZG_Face/face_test_video/原始视频/shenzhen_video_0627_1920_1080/"
    args.video_dir = "/mnt/huanyuan2/data/image/ZG_Face/face_test_video/原始视频/shenzhen_video_0629_1920_1080/"
    # args.p3d_video_endsname = "000003507600.avi"
    # args.face_video_endsname = "000001507600.avi"
    # args.p3d_video_endsname = "01p01C000000.avi"
    # args.face_video_endsname = "01p01A000000.avi"
    # args.p3d_video_endsname = "000003507600.264.avi"
    # args.face_video_endsname = "000001507600.264.avi"
    # args.p3d_video_endsname = "000003516600.avi"
    # args.face_video_endsname = "000001516600.avi"
    args.p3d_video_endsname = "000005750000.avi"
    args.face_video_endsname = "000002750000.avi"
    args.p3d_xml_endsname = "_aigrpid40.xml"
    args.face_xml_endsname = "_aigrpid73.xml"

    # args.output_video_dir = "/home/huanyuan/temp/test_video/"
    # args.output_video_dir = "/home/huanyuan/temp/shenzhen_video_0610_1920_1080/"
    # args.output_video_dir = "/home/huanyuan/temp/shenzhen_video_0625_1920_1080/"
    # args.output_video_dir = "/home/huanyuan/temp/shenzhen_video_0627_1920_1080/"
    args.output_video_dir = "/home/huanyuan/temp/shenzhen_video_0629_1920_1080/"
    args.suffix = '.avi'
    # args.suffix = '.mp4'

    # 是否保存每一帧结果
    args.write_result_per_frame_bool = True

    inference_video(args)


if __name__ == '__main__':
    main()