import torch
import cv2
import matplotlib.pyplot as plt

from options.test_options import TestOptions
from models.models import create_model


def model_initial():
    # 初始化模型，载入训练好的参数信息
    opt = TestOptions().parse()
    label_num = 26
    model = create_model(opt, label_num)
    model.load_state_dict(torch.load(opt.test_model))
    return model


def image_process(input_path:str):
    # 将图片转换成神经网络需要的输入信息
    processed_img = plt.imread(input_path)
    processed_img = processed_img[:, :, 0]  # 转换为单通道
    processed_img = cv2.resize(processed_img, (28, 28))  # 转换为28*28尺寸
    processed_img = torch.from_numpy(processed_img)  # 转换为张量
    processed_img = torch.unsqueeze(processed_img, dim=0)  # 添加一个维度
    processed_img = torch.unsqueeze(processed_img, dim=0) / 255.  # 再添加一个维度并把灰度映射在(0,1之间)
    # print(processed_img.size())#torch.Size([1, 1, 28, 28])卷积层需要4个维度的输入
    # plt.imshow(processed_img.squeeze())
    # plt.show()
    return processed_img

def predict(model,image_process):
    predict_output = model(image_process)
    # print(predict_output)
    pred_letter_asc = int(torch.max(predict_output, 1)[1])
    return chr(pred_letter_asc + 65 + 32) # char 小写字母






