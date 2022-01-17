import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision import utils
import torchvision.transforms as T
import torch.utils.data as Data
from PIL import Image
import numpy as np
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt





class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.Conv1 = nn.Sequential(
            #卷积层1
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            #激活函数层
            nn.ReLU(inplace=True),
            #最大池化层
            nn.MaxPool2d(kernel_size = 2)
        )
        self.Conv2 = nn.Sequential(
            #卷积层2
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            #激活函数层
            nn.ReLU(inplace=True),
            #最大池化层
            nn.MaxPool2d(kernel_size = 2)
        )
        #最后接上一个全连接层(将图像变为1维)
        #为什么是32*7*7：(1,28,28)->(16,28,28)(conv1)->(16,14,14)(pool1)->(32,14,14)(conv2)->(32,7,7)(pool2)->output
        self.Linear = nn.Sequential(
            nn.Linear(32*7*7,800),
            nn.Dropout(p = 0.5),
            nn.ReLU(inplace=True),
            nn.Linear(800,160),
            nn.Dropout(p = 0.5),
            nn.ReLU(inplace=True),
            nn.Linear(160,26),
         )

    def forward(self, input):
        input = self.Conv1(input)
        input = self.Conv2(input)       #view可理解为resize
        #input.size() = [100, 32, 7, 7], 100是每批次的数量，32是厚度，图片尺寸为7*7
        #当某一维是-1时，会自动计算他的大小(原则是总数据量不变):
        input = input.view(input.size(0), -1) #(batch=100, 1568), 最终效果便是将二维图片压缩为一维(数据量不变)
        #最后接上一个全连接层，输出为10:[100,1568]*[1568,10]=[100,10]
        output = self.Linear(input)
        return output






#读取网络框架
cnn = CNN()
#读取权重：
cnn.load_state_dict(torch.load('EMNIST_CNN.pkl'))


#test_x:(10000行1列，每列元素为28*28矩阵)
# 提供自己的数据进行测试：
my_img = plt.imread("/home3/hqlab/gangroup/cswFile/CVFile/ocr_stage2/datasets/G.jpg")
my_img = my_img[:,:,0] #转换为单通道
my_img = cv2.resize(my_img,(28,28))#转换为28*28尺寸
my_img = torch.from_numpy(my_img)#转换为张量
my_img = torch.unsqueeze(my_img, dim = 0)#添加一个维度
my_img = torch.unsqueeze(my_img, dim = 0)/255. #再添加一个维度并把灰度映射在(0,1之间)
#print(my_img.size())#torch.Size([1, 1, 28, 28])卷积层需要4个维度的输入


test_output = cnn(my_img)


#
# #可视化部分：
#
# #输入原图像：
# plt.imshow(my_img.squeeze())
# plt.show()
#
#
#
# #Conv1:
# cnt = 1
# my_img = cnn.Conv1(my_img)
# img = my_img.squeeze()
# for i in img.squeeze():
#
#     plt.axis('off')
#     fig = plt.gcf()
#     fig.set_size_inches(5,5)#输出width*height像素
#     plt.margins(0,0)
#
#     plt.imshow(i.detach().numpy())
#     plt.subplot(4, 4, cnt)
#     plt.axis('off')
#     plt.imshow(i.detach().numpy())
#     cnt += 1
# plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
# plt.show()
#
#
#
# #Conv2:
# cnt = 1
# my_img = cnn.Conv2(my_img)
# img = my_img.squeeze()
# for i in img.squeeze():
#
#     plt.axis('off')
#     fig = plt.gcf()
#     fig.set_size_inches(5,5)#输出width*height像素
#     plt.margins(0,0)
#
#     plt.imshow(i.detach().numpy())
#     plt.subplot(4, 8, cnt)
#     plt.axis('off')
#     plt.imshow(i.detach().numpy())
#     cnt += 1
# #plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
# plt.show()
#
#
#
#
# #全连接层：
# my_img = my_img.view(my_img.size(0), -1)
# fig = plt.gcf()
# fig.set_size_inches(10000,4)#输出width*height像素
# plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
# plt.margins(0,0)
#
#
# my_img = cnn.Linear[0](my_img)
# plt.subplot(3, 1, 1)
# plt.imshow(my_img.detach().numpy())
#
# my_img = cnn.Linear[1](my_img)
# my_img = cnn.Linear[2](my_img)
# my_img = cnn.Linear[3](my_img)
# plt.subplot(3, 1, 2)
# plt.imshow(my_img.detach().numpy())
#
# my_img = cnn.Linear[4](my_img)
# my_img = cnn.Linear[5](my_img)
# plt.subplot(3, 1, 3)
# plt.imshow(my_img.detach().numpy())
#
# # plt.show()
#


#输出预测结果:
pred_y = int(torch.max(test_output,1)[1])
#chr()将数字转为对应的的ASCAII字符
print('\npredict character: %c or %c' % (chr(pred_y+65),chr(pred_y+97)))
