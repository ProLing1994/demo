
import clip
from PIL import Image
import torch


img_pah = "/yuanhuan/data/image/Taxi_remnant/sd_crop_0810/shenzhen/Pickup_middle_20230721/references/256_256/remnant_pickup_middle_shenzhen_20230721_0001_remnants_0.jpg"
img_ref_pah_list = ["/yuanhuan/data/image/Taxi_remnant/sd_crop_0810/shenzhen/Pickup_middle_20230721/references/512_512/remnant_pickup_middle_shenzhen_20230721_0311_remnants_0.jpg", 
                    "/yuanhuan/data/image/Taxi_remnant/sd_crop_0810/shenzhen/Pickup_middle_20230721/references/512_512/remnant_pickup_middle_shenzhen_20230721_2072_remnants_0.jpg"]
classes = ['flask', 'bottle', 'cup', 'not_cup']

#加载模型
print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

#准备输入集
image = Image.open(img_pah)
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device) #生成文字描述
image_ref_input_list = []
for idx in range(len(img_ref_pah_list)):
    image_ref = Image.open(img_ref_pah_list[idx])
    image_ref_input = preprocess(image_ref).unsqueeze(0).to(device)
    image_ref_input_list.append(image_ref_input)
image_ref_input = torch.cat(image_ref_input_list).to(device)

#特征编码
with torch.no_grad():
    image_features = model.encode_image(image_input)
    image_ref_features = model.encode_image(image_ref_input)
    text_features = model.encode_text(text_inputs)

#选取参数最高的标签
image_features /= image_features.norm(dim=-1, keepdim=True)
image_ref_features /= image_ref_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1) #对图像描述和图像特征  
similarity_ref = (100.0 * image_features @ image_ref_features.T).softmax(dim=-1) #对图像描述和图像特征  
# values, indices = similarity[0].topk(1)

# #输出结果
# print("\nTop predictions:\n")
# print()
values, indices = similarity[0].topk(3)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print('classes:{} score:{:.2f}'.format(classes[index], value.item()))
print()