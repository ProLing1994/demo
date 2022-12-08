import cv2
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET

def gen_mosaic(image, bboxes_list, mosaic_size = 5):
  for box_id in range(len(bboxes_list)):  
    x1 = bboxes_list[box_id]["bbox"][0]
    y1 = bboxes_list[box_id]["bbox"][1]
    x2 = bboxes_list[box_id]["bbox"][2]
    y2 = bboxes_list[box_id]["bbox"][3]
    w = x2 - x1
    h = y2 - y1
    
    for idx in range(0, w, mosaic_size):
      for idy in range(0, h, mosaic_size):  

        mosaic_size_w = mosaic_size
        mosaic_size_h = mosaic_size
        
        if w - idx <= mosaic_size:
          mosaic_size_w = w - idx
        if h - idy <= mosaic_size:
          mosaic_size_h = h - idy
          
        rect = [idx + x1, idy + y1, mosaic_size_w, mosaic_size_h]
        color = image[rect[1]][rect[0]].tolist()  
        left_up = (rect[0], rect[1])
        right_down = (rect[0] + rect[2] - 1, rect[1] + rect[3] - 1)  
        cv2.rectangle(image, left_up, right_down, color, -1)


if __name__ == "__main__":
  # input_dir = "/home/huanyuan/data/ceiba_mosaic/bus/20200918/check/"
  # output_dir = "/home/huanyuan/data/ceiba_mosaic/bus/20200918/"
  input_dir = "/home/huanyuan/data/ceiba_mosaic/bus_002/"
  output_dir = "/home/huanyuan/data/ceiba_mosaic/bus_002/"

  data_list = os.listdir(input_dir)
  image_list = [data for data in data_list if data.endswith(".jpg")]
  label_list = [data for data in data_list if data.endswith(".xml")]
  image_list.sort()

  fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
  # video_writer = cv2.VideoWriter(os.path.join(output_dir, "bus_mosaic.avi"), fourcc, 15, (1280, 720))
  video_writer = cv2.VideoWriter(os.path.join(output_dir, "bus_002_mosaic.avi"), fourcc, 14, (704, 576))

  for idx in tqdm(range(len(image_list))):
    image_path = os.path.join(input_dir, image_list[idx])
    print("image_path：{}".format(image_path))

    image = cv2.imread(image_path)  
    image_width = image.shape[1]
    image_heght = image.shape[0]
    if (image_list[idx].replace(".jpg", ".xml") in label_list):
      tree = ET.parse(os.path.join(input_dir, image_list[idx].replace(".jpg", ".xml")))
      root = tree.getroot()

      size = root.find('size')
      w = int(size.find('width').text)
      h = int(size.find('height').text)
      d = int(size.find('depth').text)
      assert image_width == w
      assert image_heght == h

      # add box 
      bboxes_list = []
      for obj in root.findall('object'):
        bboxes_subdict = {}
        bboxes_subdict["name"]  = obj.find('name').text
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        bboxes_subdict["bbox"] = bbox
        bboxes_list.append(bboxes_subdict)
      
      # mosaic
      gen_mosaic(image, bboxes_list)

      # retangle 
      for box_id in range(len(bboxes_list)):
        x1 = bboxes_list[box_id]["bbox"][0]
        y1 = bboxes_list[box_id]["bbox"][1]
        x2 = bboxes_list[box_id]["bbox"][2]
        y2 = bboxes_list[box_id]["bbox"][3]
        cv2.rectangle(image, (x1,y1), (x2,y2), (0, 0, 255), 2) 

    video_writer.write(image)
    # cv2.imshow("image", image)
    # cv2.waitKey(1)
    # print()
  video_writer.release()
  print("Done")