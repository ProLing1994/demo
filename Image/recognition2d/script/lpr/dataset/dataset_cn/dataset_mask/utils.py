import cv2
import numpy as np

def draw_mask(dataset_dict, img_crop, object_roi, output_img_path, output_mask_path, output_mask_img_path, output_bbox_img_path):
        
        # mask mask_img bbox_img
        mask = np.zeros(img_crop.shape, dtype=img_crop.dtype)
        mask_img = np.zeros(img_crop.shape, dtype=img_crop.dtype)
        bbox_img = img_crop.copy()

        # 按照一定顺序画图
        # 目的：存在重叠区域
        for idy in range(len(dataset_dict.add_mask_order)):
            class_name_key = dataset_dict.add_mask_order[idy]
            for idz in range(len(object_roi)):
                classname = object_roi[idz]["classname"]
                bndbox = object_roi[idz]["bndbox"]

                if classname != class_name_key:
                    continue

                contours = []
                contours.append([bndbox[0], bndbox[1]])
                contours.append([bndbox[2], bndbox[1]])
                contours.append([bndbox[2], bndbox[3]])
                contours.append([bndbox[0], bndbox[3]])
                contours = [np.array(contours).reshape(-1, 1, 2)]

                mask = cv2.drawContours(mask, contours, -1, dataset_dict.name_2_mask_id_dict[classname], cv2.FILLED)
                mask_img = cv2.drawContours(mask_img, contours, -1, dataset_dict.name_2_mask_color_dict[classname], cv2.FILLED)
                bbox_img = cv2.rectangle(bbox_img, (bndbox[0], bndbox[1]), (bndbox[2], bndbox[3]), color=dataset_dict.name_2_mask_color_dict[classname], thickness=2)

        # mask_img
        mask_img = cv2.addWeighted(src1=img_crop, alpha=0.8, src2=mask_img, beta=0.3, gamma=0.)

        cv2.imwrite(output_mask_path, mask)
        cv2.imwrite(output_mask_img_path, mask_img)
        cv2.imwrite(output_bbox_img_path, bbox_img)
        cv2.imwrite(output_img_path, img_crop)