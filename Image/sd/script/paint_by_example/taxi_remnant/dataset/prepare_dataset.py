import argparse
import clip
import cv2
import numpy as np
import os
from PIL import Image
import sys 
import shutil
import torch
from tqdm import tqdm
import xml.etree.ElementTree as ET

sys.path.insert(0, '/yuanhuan/code/demo/Image')
from Basic.utils.folder_tools import *
from Basic.script.xml.xml_write import write_xml


sys.path.insert(0, '/yuanhuan/code/demo/Image/segmentation2d/segment-anything/')
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator


def center_crop(args, img, bndbox):
    
    crop_width = bndbox[2] - bndbox[0]
    crop_height = bndbox[3] - bndbox[1]
    crop_cener_x = (bndbox[0] + bndbox[2]) / 2
    crop_cener_y = (bndbox[1] + bndbox[3]) / 2
    
    crop_size = (720, 720)
    for idy in range(len(args.crop_size_list)):
        
        if crop_width < args.crop_size_list[idy][0] * 0.6 and crop_height < args.crop_size_list[idy][1] * 0.6 :
            crop_size = args.crop_size_list[idy]
            break

    # roi_crop
    roi_crop = [0, 0, 0, 0]
    roi_crop[0] = crop_cener_x - crop_size[0] / 2
    roi_crop[1] = crop_cener_y - crop_size[1] / 2
    roi_crop[2] = crop_cener_x + crop_size[0] / 2
    roi_crop[3] = crop_cener_y + crop_size[1] / 2

    if roi_crop[0] < 0:
        transform_x = 0 - roi_crop[0]
        roi_crop[0] = 0
        roi_crop[2] += transform_x
    if roi_crop[1] < 0:
        transform_y = 0 - roi_crop[1]
        roi_crop[1] = 0
        roi_crop[3] += transform_y
    if roi_crop[2] > img.shape[1]:  
        transform_x = roi_crop[2] - img.shape[1]
        roi_crop[2] = img.shape[1] - 1
        roi_crop[0] -= transform_x + 1
    if roi_crop[3] > img.shape[0]:
        transform_y = roi_crop[3] - img.shape[0]
        roi_crop[3] = img.shape[0] - 1
        roi_crop[1] -= transform_y + 1

    roi_crop[0] = int(roi_crop[0] + 0.5)
    roi_crop[1] = int(roi_crop[1] + 0.5)
    roi_crop[2] = int(roi_crop[2] + 0.5)
    roi_crop[3] = int(roi_crop[3] + 0.5)

    return roi_crop


def clip_init(args):
    # CLIP 加载模型
    print(clip.available_models())
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.model, args.preprocess = clip.load('/yuanhuan/model/image/clip/clip/ViT-B-32.pt', args.device)
    # args.model, args.preprocess = clip.load('ViT-B/32', args.device)

    # CLIP 准备输入集
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in args.clip_class_list]).to(args.device) #生成文字描述

    # CLIP 特征编码
    with torch.no_grad():
        args.text_features =args.model.encode_text(text_inputs)

    return


def clip_ref_value(args, img_crop, output_tmp_img_path):
    
    cv2.imwrite(output_tmp_img_path, img_crop)

    # CLIP 准备输入集
    image = Image.open(output_tmp_img_path)
    image_input = args.preprocess(image).unsqueeze(0).to(args.device)
    
    # CLIP 特征编码
    with torch.no_grad():
        image_features = args.model.encode_image(image_input)

    # CLIP 选取参数最高的标签
    image_features /= image_features.norm(dim=-1, keepdim=True)
    args.text_features /= args.text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ args.text_features.T).softmax(dim=-1) #对图像描述和图像特征 
    values, indices = similarity[0].topk(5)

    ref_value = 0.0
    for value, index in zip(values, indices):
        if args.clip_class_list[index] in args.clip_ref_class_list:
            ref_value += value.item()
    
    return ref_value


def sam_init(args):

    sam_checkpoint = "/yuanhuan/model/image/seg/sam/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    device = "cuda"

    # sam
    args.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    args.sam.to(device=device)

    return


def sam_predict(args, image, bbox):

    # predictor
    predictor = SamPredictor(args.sam)
    predictor.set_image(image)

    # bbox
    input_box = np.array(bbox)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False,
    )

    return masks


def prepare_dataset(args):

    # init 
    if args.clip_bool:
        clip_init(args)
    if args.sam_bool:
        sam_init(args)

    # mkdir 
    create_folder(args.output_img_dir)
    create_folder(args.output_xml_dir)
    create_folder(args.output_mask_dir)
    create_folder(args.output_mask_img_dir)
    create_folder(args.output_reference_dir)
    create_folder(args.output_img_resize_dir)
    create_folder(args.output_reference_resize_dir)
    create_folder(args.output_mask_resize_dir)
    create_folder(args.output_mask_img_resize_dir)
    create_folder(args.output_mask_sam_dir)
    create_folder(args.output_mask_img_sam_dir)
    create_folder(args.output_mask_sam_resize_dir)
    create_folder(args.output_mask_img_sam_resize_dir)
    create_folder(args.output_mask_sam_padding_resize_dir)
    create_folder(args.output_mask_img_sam_padding_resize_dir)

    # jpg list
    jpg_list = np.array(os.listdir(args.input_img_dir))
    jpg_list = jpg_list[[jpg.endswith('.jpg') for jpg in jpg_list]]
    jpg_list = jpg_list[[os.path.exists(os.path.join(args.input_xml_dir, jpg.replace(".jpg", ".xml"))) for jpg in jpg_list]]
    jpg_list.sort()

    for idx in tqdm(range(len(jpg_list))):
    
        img_path = os.path.join(args.input_img_dir, jpg_list[idx])
        xml_path = os.path.join(args.input_xml_dir, jpg_list[idx].replace(".jpg", ".xml"))

        # img
        img = cv2.imread(img_path)

        # xml
        tree = ET.parse(xml_path)  # ET是一个 xml 文件解析库，ET.parse（）打开 xml 文件，parse--"解析"
        root = tree.getroot()   # 获取根节点

        # init 
        id = 0

        # 标签检测和标签转换
        for object in root.findall('object'):
            # name
            classname = str(object.find('name').text)

            # bbox
            bbox = object.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                bndbox.append(cur_pt)
            bndbox[0] = max(0, bndbox[0])
            bndbox[1] = max(0, bndbox[1])
            bndbox[2] = min(img.shape[1], bndbox[2])
            bndbox[3] = min(img.shape[0], bndbox[3])

            # select
            if classname not in args.select_key_list:
                print(classname)
                continue

            img_name = jpg_list[idx].replace(".jpg", "")
            output_tmp_img_path = os.path.join(args.output_dir, 'tmp.jpg')
            output_img_path = os.path.join(args.output_img_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_xml_path = os.path.join(args.output_xml_dir, img_name + '_' + classname+ '_' + str(id) + '.xml')
            output_mask_path = os.path.join(args.output_mask_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_mask_img_path = os.path.join(args.output_mask_img_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_img_resize_path = os.path.join(args.output_img_resize_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_reference_resize_path = os.path.join(args.output_reference_resize_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_mask_resize_path = os.path.join(args.output_mask_resize_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_mask_img_resize_path = os.path.join(args.output_mask_img_resize_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_mask_sam_path = os.path.join(args.output_mask_sam_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_mask_img_sam_path = os.path.join(args.output_mask_img_sam_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_mask_sam_resize_path = os.path.join(args.output_mask_sam_resize_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_mask_img_sam_resize_path = os.path.join(args.output_mask_img_sam_resize_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_mask_sam_padding_resize_path = os.path.join(args.output_mask_sam_padding_resize_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')
            output_mask_img_sam_padding_resize_path = os.path.join(args.output_mask_img_sam_padding_resize_dir, img_name + '_' + classname+ '_' + str(id) + '.jpg')

            # # test
            # if img_name != "remnant_pickup_middle_shenzhen_20230721_8183":
            #     continue

            if args.crop_bool:
                # center_crop
                roi_crop = center_crop(args, img, bndbox)
                img_crop = img[roi_crop[1]:roi_crop[3], roi_crop[0]:roi_crop[2]]

                if args.clip_bool:
                    # clip_ref_value
                    ref_value = clip_ref_value(args, img_crop, output_tmp_img_path)

                    # # test
                    # if img_name == "remnant_pickup_middle_shenzhen_20230721_8183":
                    #     ref_value = 0.5

                    if ref_value > 0.3:
                        
                        # img_crop
                        cv2.imwrite(output_img_path, img_crop)
                        
                        # bndbox_crop
                        bndbox_crop = []
                        bndbox_crop.append(bndbox[0] - roi_crop[0])
                        bndbox_crop.append(bndbox[1] - roi_crop[1])
                        bndbox_crop.append(bndbox[2] - roi_crop[0])
                        bndbox_crop.append(bndbox[3] - roi_crop[1])

                        # xml
                        xml_bboxes = {}
                        xml_bboxes[args.clip_ref_class] = []   
                        xml_bboxes[args.clip_ref_class].append([bndbox_crop[0], bndbox_crop[1], bndbox_crop[2], bndbox_crop[3]])
                        write_xml(output_xml_path, output_img_path, xml_bboxes, img_crop.shape)

                        # contour_crop
                        contour_crop = []
                        contour_crop.append([bndbox_crop[0], bndbox_crop[1]])
                        contour_crop.append([bndbox_crop[2], bndbox_crop[1]])
                        contour_crop.append([bndbox_crop[2], bndbox_crop[3]])
                        contour_crop.append([bndbox_crop[0], bndbox_crop[3]])
                        contour_crop = [np.array(contour_crop).reshape(-1, 1, 2)]

                        # mask_crop
                        mask_crop = np.zeros(img_crop.shape, dtype=img_crop.dtype)
                        mask_crop = cv2.drawContours(mask_crop, contour_crop, -1, (255, 255, 255), cv2.FILLED)
                        cv2.imwrite(output_mask_path, mask_crop)

                        # mask_img_crop
                        mask_img_crop = np.zeros(img_crop.shape, dtype=img_crop.dtype)
                        mask_img_crop = cv2.drawContours(mask_img_crop, contour_crop, -1, (0, 0, 255), cv2.FILLED)
                        mask_img_crop = cv2.addWeighted(src1=img_crop, alpha=0.8, src2=mask_img_crop, beta=0.3, gamma=0.)
                        cv2.imwrite(output_mask_img_path, mask_img_crop)

                        # reference_img_crop
                        reference_img_crop = img_crop[max(0, bndbox_crop[1]-args.reference_crop_expand_pixel):min(bndbox_crop[3]+args.reference_crop_expand_pixel, img_crop.shape[0]), max(0, bndbox_crop[0]-args.reference_crop_expand_pixel):min(bndbox_crop[2]+args.reference_crop_expand_pixel, img_crop.shape[1])]
                        output_reference_path = os.path.join(args.output_reference_dir, '{}_{}'.format(img_crop.shape[0], img_crop.shape[1]), img_name + '_' + classname+ '_' + str(id) + '.jpg')
                        create_folder(os.path.dirname(output_reference_path))
                        cv2.imwrite(output_reference_path, reference_img_crop)

                        # img_crop_resize
                        bndbox_crop_width = bndbox_crop[2] - bndbox_crop[0]
                        bndbox_crop_height = bndbox_crop[3] - bndbox_crop[1]
                        bndbox_crop_size = bndbox_crop_width * bndbox_crop_height
                        bndbox_crop_size_ratio = bndbox_crop_size / (img_crop.shape[0] * img_crop.shape[1])
                        # 参考官方处理代码
                        if bndbox_crop_size_ratio > 0.8 or bndbox_crop_size_ratio < 0.02:
                            continue
                        img_crop_resize = cv2.resize(img_crop, (args.resize_data_size[0], args.resize_data_size[1]), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(output_img_resize_path, img_crop_resize)

                        # bndbox_crop_resize
                        bndbox_crop_resize = []
                        crop_reszie_ratio = (args.resize_data_size[0] / (img_crop.shape[0]), args.resize_data_size[1] / (img_crop.shape[1]))
                        bndbox_crop_resize.append(int(bndbox_crop[0] * crop_reszie_ratio[0]))
                        bndbox_crop_resize.append(int(bndbox_crop[1] * crop_reszie_ratio[1]))
                        bndbox_crop_resize.append(int(bndbox_crop[2] * crop_reszie_ratio[0]))
                        bndbox_crop_resize.append(int(bndbox_crop[3] * crop_reszie_ratio[1]))

                        # contour_crop_resize
                        contour_crop_resize = []
                        contour_crop_resize.append([bndbox_crop_resize[0], bndbox_crop_resize[1]])
                        contour_crop_resize.append([bndbox_crop_resize[2], bndbox_crop_resize[1]])
                        contour_crop_resize.append([bndbox_crop_resize[2], bndbox_crop_resize[3]])
                        contour_crop_resize.append([bndbox_crop_resize[0], bndbox_crop_resize[3]])
                        contour_crop_resize = [np.array(contour_crop_resize).reshape(-1, 1, 2)]

                        # mask_crop_resize
                        mask_crop_resize = np.zeros(img_crop_resize.shape, dtype=img_crop_resize.dtype)
                        mask_crop_resize = cv2.drawContours(mask_crop_resize, contour_crop_resize, -1, (255, 255, 255), cv2.FILLED)
                        cv2.imwrite(output_mask_resize_path, mask_crop_resize)

                        # mask_img_crop_resize
                        mask_img_crop_resize = np.zeros(img_crop_resize.shape, dtype=img_crop_resize.dtype)
                        mask_img_crop_resize = cv2.drawContours(mask_img_crop_resize, contour_crop_resize, -1, (0, 0, 255), cv2.FILLED)
                        mask_img_crop_resize = cv2.addWeighted(src1=img_crop_resize, alpha=0.8, src2=mask_img_crop_resize, beta=0.3, gamma=0.)
                        cv2.imwrite(output_mask_img_resize_path, mask_img_crop)

                        # reference_img_resize_crop
                        reference_img_resize_crop = img_crop_resize[max(0, bndbox_crop_resize[1]-args.reference_crop_expand_pixel):min(bndbox_crop_resize[3]+args.reference_crop_expand_pixel, img_crop_resize.shape[0]), max(0, bndbox_crop_resize[0]-args.reference_crop_expand_pixel):min(bndbox_crop_resize[2]+args.reference_crop_expand_pixel, img_crop_resize.shape[1])]
                        cv2.imwrite(output_reference_resize_path, reference_img_resize_crop)

                        # sam
                        if args.sam_bool:
                            
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            sam_mask = sam_predict(args, img, bndbox)
                            sam_mask = np.squeeze(sam_mask.astype(np.uint8))
                            contours, hierarchy = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # contour_crop
                            contour_crop = []
                            contour_crop.append(contours[0])
                            contour_crop[0][:, :, 0] = contour_crop[0][:, :, 0] - roi_crop[0]
                            contour_crop[0][:, :, 1] = contour_crop[0][:, :, 1] - roi_crop[1]

                            # mask_crop
                            mask_crop = np.zeros(img_crop.shape, dtype=img_crop.dtype)
                            mask_crop = cv2.drawContours(mask_crop, contour_crop, -1, (255, 255, 255), cv2.FILLED)
                            cv2.imwrite(output_mask_sam_path, mask_crop)

                            # mask_img_crop
                            mask_img_crop = np.zeros(img_crop.shape, dtype=img_crop.dtype)
                            mask_img_crop = cv2.drawContours(mask_img_crop, contour_crop, -1, (0, 0, 255), cv2.FILLED)
                            mask_img_crop = cv2.addWeighted(src1=img_crop, alpha=0.8, src2=mask_img_crop, beta=0.3, gamma=0.)
                            cv2.imwrite(output_mask_img_sam_path, mask_img_crop)

                            # contour_crop_resize
                            contour_crop_resize = []
                            contour_crop_resize.append(contour_crop[0])
                            contour_crop_resize[0][:, :, 0] = contour_crop_resize[0][:, :, 0] * crop_reszie_ratio[0]
                            contour_crop_resize[0][:, :, 1] = contour_crop_resize[0][:, :, 1] * crop_reszie_ratio[1]
                            contour_crop_resize[0] = contour_crop_resize[0].astype(np.int32)
                        
                            # mask_sam_resize
                            mask_sam_resize = np.zeros(img_crop_resize.shape, dtype=img_crop_resize.dtype)
                            mask_sam_resize = cv2.drawContours(mask_sam_resize, contour_crop_resize, -1, (255, 255, 255), cv2.FILLED)
                            cv2.imwrite(output_mask_sam_resize_path, mask_sam_resize)

                            # mask_img_sam_resize
                            mask_img_sam_resize = np.zeros(img_crop_resize.shape, dtype=img_crop_resize.dtype)
                            mask_img_sam_resize = cv2.drawContours(mask_img_sam_resize, contour_crop_resize, -1, (0, 0, 255), cv2.FILLED)
                            mask_img_sam_resize_out = cv2.addWeighted(src1=img_crop_resize, alpha=0.8, src2=mask_img_sam_resize, beta=0.3, gamma=0.)
                            cv2.imwrite(output_mask_img_sam_resize_path, mask_img_sam_resize_out)

                            # mask_sam_padding_resize
                            kernel = np.ones((5, 5), np.uint8) 
                            # mask_sam_padding_resize = cv2.dilate(mask_sam_resize, kernel, iterations = 10)
                            mask_sam_padding_resize = cv2.dilate(mask_sam_resize, kernel, iterations = 5)
                            cv2.imwrite(output_mask_sam_padding_resize_path, mask_sam_padding_resize)

                            # mask_sam_padding_resize
                            kernel = np.ones((5, 5), np.uint8) 
                            # mask_img_sam_padding_resize = cv2.dilate(mask_img_sam_resize, kernel, iterations = 10)
                            mask_img_sam_padding_resize = cv2.dilate(mask_img_sam_resize, kernel, iterations = 5)
                            mask_img_sam_padding_resize = cv2.addWeighted(src1=img_crop_resize, alpha=0.8, src2=mask_img_sam_padding_resize, beta=0.3, gamma=0.)
                            cv2.imwrite(output_mask_img_sam_padding_resize_path, mask_img_sam_padding_resize)

                        id += 1


def check_dataset(args):

    # mask list
    mask_list = np.array(os.listdir(args.output_mask_sam_resize_dir))
    mask_list = mask_list[[jpg.endswith('.jpg') for jpg in mask_list]]
    mask_list.sort()

    for idx in tqdm(range(len(mask_list))):
        
        mask_name = mask_list[idx]
        mask_path = os.path.join(args.output_mask_sam_resize_dir, mask_name)
        
        # mask
        mask = cv2.imread(mask_path)
        mask_num = (mask[:,:,0] == 255).sum()
        
        # 如果 mask 太小，认为 sam 结果不合适，删掉图像
        if mask_num < args.mask_threh:

            output_mask_sam_path = os.path.join(args.output_mask_sam_dir, mask_name)
            output_mask_img_sam_path = os.path.join(args.output_mask_img_sam_dir, mask_name)
            output_mask_sam_resize_path = os.path.join(args.output_mask_sam_resize_dir, mask_name)
            output_mask_img_sam_resize_path = os.path.join(args.output_mask_img_sam_resize_dir, mask_name)
            output_mask_sam_padding_resize_path = os.path.join(args.output_mask_sam_padding_resize_dir, mask_name)
            output_mask_img_sam_padding_resize_path = os.path.join(args.output_mask_img_sam_padding_resize_dir, mask_name)
            
            print(output_mask_sam_path)
            os.remove(output_mask_sam_path)

            print(output_mask_img_sam_path)
            os.remove(output_mask_img_sam_path)

            print(output_mask_sam_resize_path)
            os.remove(output_mask_sam_resize_path)

            print(output_mask_img_sam_resize_path)
            os.remove(output_mask_img_sam_resize_path)

            print(output_mask_sam_padding_resize_path)
            os.remove(output_mask_sam_padding_resize_path)

            print(output_mask_img_sam_padding_resize_path)
            os.remove(output_mask_img_sam_padding_resize_path)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--date_name', type=str, default="Pickup_middle_20230721") 
    parser.add_argument('--input_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/original/shenzhen") 
    parser.add_argument('--output_dir', type=str, default="/yuanhuan/data/image/Taxi_remnant/training/sd_crop_clip_sam_bottle_all_0825/shenzhen") 
    parser.add_argument('--clip_ref_class', type=str, default='bottle') 
    args = parser.parse_args()

    args.input_dir = os.path.join(args.input_dir, args.date_name)
    args.output_dir = os.path.join(args.output_dir, args.date_name)

    print("prepare dataset.")
    print("date_name: {}".format(args.date_name))
    print("input_dir: {}".format(args.input_dir))
    print("output_dir: {}".format(args.output_dir))

    args.input_img_dir = os.path.join(args.input_dir, 'image')
    args.input_xml_dir = os.path.join(args.input_dir, 'xml')
    args.output_img_dir = os.path.join(args.output_dir, 'JPEGImages')
    args.output_xml_dir = os.path.join(args.output_dir, 'Annotations')
    args.output_img_resize_dir = os.path.join(args.output_dir, 'JPEGImages_resize')
    args.output_reference_dir = os.path.join(args.output_dir, 'references')
    args.output_mask_dir = os.path.join(args.output_dir, 'masks')
    args.output_mask_img_dir = os.path.join(args.output_dir, 'mask_imgs')
    args.output_mask_sam_dir = os.path.join(args.output_dir, 'masks_sam')
    args.output_mask_img_sam_dir = os.path.join(args.output_dir, 'mask_imgs_sam')
    args.output_reference_resize_dir = os.path.join(args.output_dir, 'references_resize')
    args.output_mask_resize_dir = os.path.join(args.output_dir, 'masks_resize')
    args.output_mask_img_resize_dir = os.path.join(args.output_dir, 'mask_imgs_resize')
    args.output_mask_sam_resize_dir = os.path.join(args.output_dir, 'masks_sam_resize')
    args.output_mask_img_sam_resize_dir = os.path.join(args.output_dir, 'mask_imgs_sam_resize')
    # args.output_mask_sam_padding_resize_dir = os.path.join(args.output_dir, 'masks_sam_resize_padding')
    # args.output_mask_img_sam_padding_resize_dir = os.path.join(args.output_dir, 'mask_imgs_sam_resize_padding')
    args.output_mask_sam_padding_resize_dir = os.path.join(args.output_dir, 'masks_sam_resize_padding_5')
    args.output_mask_img_sam_padding_resize_dir = os.path.join(args.output_dir, 'mask_imgs_sam_resize_padding_5')

    # select
    args.select_key_list = ['remnants']

    # crop
    args.crop_bool = True
    args.crop_size_list = [(64, 64), (128, 128), (256, 256), (512, 512), (720, 720)]    # w, h
    
    # clip
    args.clip_bool = True
    args.clip_class_list = ['flask', 'bottle', 'cup', 'cigarette', 'phone', 'wallet', 'power', 'umbrella', 'bag', 'cloth', "helmet"]
    if args.clip_ref_class == "bottle":
        args.clip_ref_class_list = ['flask', 'bottle', 'cup']
    elif args.clip_ref_class == "wallet":
        args.clip_ref_class_list = ['wallet']
    elif args.clip_ref_class == "umbrella":
        args.clip_ref_class_list = ['umbrella']
    else:
        raise Exception

    # sam
    args.sam_bool = True

    # reference
    args.reference_crop_expand_pixel = 10

    # resize
    args.resize_data_size = (512, 512)
    
    prepare_dataset(args)

    # 过滤不合理的 mask
    args.mask_threh = args.resize_data_size[0]*args.resize_data_size[1]*0.02
    check_dataset(args)