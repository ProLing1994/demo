import glob
import os
import shutil
from tqdm import tqdm 

if __name__ == '__main__':
  input_dir = "/home/huanyuan/data/images/Face/busface/"
  output_dir = "/home/huanyuan/data/images/Face/busface_total"

  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  
  image_list = glob.glob(os.path.join(input_dir, '*/*/' +'*.jpg'))
  total_image_list = []
  total_image_path_list = []
  for idx in tqdm(range(len(image_list))):
    print(image_list[idx])
    image_name = os.path.basename(image_list[idx])
    folder_name = os.path.basename(os.path.dirname(image_list[idx]))
    mask_name = os.path.basename(os.path.dirname(os.path.dirname(image_list[idx])))
    image_name = mask_name + '_' + folder_name + '_' + image_name
    assert image_name not in total_image_list, "{}".format(image_list[idx])

    total_image_list.append(image_name)
    total_image_path_list.append(image_list[idx])
    shutil.copy(image_list[idx], os.path.join(output_dir, image_name))
  print()