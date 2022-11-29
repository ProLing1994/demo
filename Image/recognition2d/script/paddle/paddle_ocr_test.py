from paddleocr import PaddleOCR

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="en")  # need to run only once to download and load model into memory

# img_path = '/mnt/huanyuan2/data/image/ZD_anpr/ppocr_img/ppocr_img/imgs/11.jpg'
# img_path = "/mnt/huanyuan/temp/data/plate_zd_mask/data_crop_0828_0831/Images/0000000000000000-220828-180130-180231-00000F000180_54_18_60_96_00090-00_none_none_none_Single_8#86661.jpg"
# img_path = "/mnt/huanyuan/temp/data/plate_zd_mask/data_crop_0828_0831/Images/0000000000000000-220828-180130-180233-00000G000180_23_37_47_21_00100-00_none_none_none_Single_14#51248.jpg"
img_path = "/mnt/huanyuan/temp/data/plate_zd_mask/data_crop_0828_0831/Images/0000000000000000-220828-180130-180233-00000G000180_23_37_47_21_00470-00_none_none_none_Single_B#9633.jpg"

# result = ocr.ocr(img_path, cls=True)
result = ocr.ocr(img_path, cls=True, det=False)

for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)