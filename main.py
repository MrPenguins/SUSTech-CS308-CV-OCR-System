import torch
import torch.nn as nn
# torchvision已经预先实现了常用的Datast
from torchvision.datasets import ImageFolder # ImageFolder是一个经常用到的Dataset
import torchvision.models as models
from torchvision import utils
import torchvision.transforms as T
import torch.utils.data as Data
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import matplotlib.pyplot as plt
#使用tensorboardX进行可视化
from tensorboardX import SummaryWriter

# 创建一个write实例，自动生成的文件夹路径为：./EMNIST_log
SumWriter = SummaryWriter(log_dir = "./EMNIST_log")


# 数据预处理
# 首先定义超参数
EPOCH = 10   # 训练的批次，下载的数据集每个字母都有5000张左右的图片，由于电脑性能的原因，对于每个字母的训练我只保留了1000张图片，同时为了保证训练准确度，将训练的次数调得比较多
BATCH_SIZE = 128    # 训练的最小规模（一次反向传播需要更新权值）
LR = 1e-4   # 学习率


# 转为tensor 以及 标准化
transform = T.Compose([
     #转为灰度图像，这部分是便于图像识别的:
     T.Grayscale(num_output_channels=1),
     #将图片转换为Tensor,归一化至(0,1)，在实验中发现如果没有归一化的过程，最后的预测效果会很差:
     T.ToTensor(),
])


#数据集要作为一个文件夹读入：
# #读取训练集：

# ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
# 它主要有四个参数：
# root：在root指定的路径下寻找图片
# transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
# target_transform：对label的转换
# loader：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象
train_data = ImageFolder(root="./emnist-chars74k_datasets/Train_png",
                         transform=transform)

# 训练集数据的加载器，自动将数据分割成batch，顺序随机打乱
# shuffle这个参数代表是否在构建批次时随机选取数据
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

#读取测试集：
test_data = ImageFolder(root="./emnist-chars74k_datasets/Test_png",
                        transform=transform)
#之所以要将test_data转换为loader是因为网络不支持原始的ImageFolder类数据，到时候直接使用批训练，便是tensor类。
#batch_size为全部10000张testdata，在全测试集上测试精度
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=test_data.__len__())
label_num = len(train_data.class_to_idx)


#数据可视化：
to_img = T.ToPILImage()
a=to_img(test_data[0][0]) #size=[1, 28, 28]
plt.imshow(a)
plt.axis('off')
plt.show()

# 卷积网络搭建：两层卷积网络（卷积+池化）+ 三层全连接层
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Sequantial()把不同的函数组合成一个模块使用
        # 定义网络框架
        # 卷积层1（卷积核=16）
        self.Conv1 = nn.Sequential(
            # 5个参数依次是：
            # in_channels:输入图像的通道数，这里为1表示只有一层图像
            # out_channels:定义了16个卷积核，即输出高度为16
            # kernel_size:卷积核大小为5 * 5
            # stride: 步长，卷积核每次扫描的跨度
            # padding: 边界填充0（如步长为1时,若要保证输出尺寸像和原尺寸一致,
            #           计算公式为:padding = (kernel_size-1)/2）
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            #激活函数层
            nn.ReLU(),
            #最大池化层 通过最大值进行池化
            nn.MaxPool2d(kernel_size=2)
        )
        # 卷积层2
        self.Conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            #激活函数层
            nn.Dropout(p=0.2),
            nn.ReLU(),
            #最大池化层
            nn.MaxPool2d(kernel_size=2)
        )

        #最后接上一个全连接层(将图像变为1维)
        #为什么是32*7*7：
        # (1,28,28)->(16,28,28)(conv1)
        # ->(16,14,14)(pool1)
        # ->(32,14,14)(conv2)
        # ->(32,7,7)(pool2)->output
        self.Linear = nn.Sequential(
            nn.Linear(32*7*7,400),
            # Dropout按指定概率随机舍弃部分的神经元
            nn.Dropout(p = 0.5),
            # 全连接层激活函数
            nn.ReLU(),
            nn.Linear(400,80),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(80,label_num),
         )

    # 前向传播
    def forward(self, input):
        input = self.Conv1(input)
        input = self.Conv2(input)
        #input.size() = [100, 32, 7, 7], 100是每批次的数量，32是厚度，图片尺寸为7*7
        #当某一维是-1时，会自动计算它的大小(原则是总数据量不变):
        input = input.view(input.size(0), -1) #(batch=100, 1568), 最终效果便是将二维图片压缩为一维(数据量不变)
        #最后接上一个全连接层，输出为10:[100,1568]*[1568,10]=[100,10]
        output = self.Linear(input)
        return output


# 读取网络框架
cnn = CNN()
# 仅保存训练好的参数
torch.save(cnn.state_dict(), 'EMNIST_CNN.pkl')
# 加载训练好的参数
cnn.load_state_dict(torch.load('EMNIST_CNN.pkl'))
# 进行训练
cnn.train()
# 显示网络层结构
# print(cnn)

#定义优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
#定义损失函数，因为是分类问题，所以使用交叉熵损失
loss_func = nn.CrossEntropyLoss()

# 训练与模式保存
# 根据EPOCH自动更新学习率，2次EPOCH学习率减少为原来的一半:
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma = 0.6, last_epoch = -1)

for epoch in range(EPOCH):
    # enumerate() 函数用于将一个可遍历的数据对象组合为一个索引序列。例:['A','B','C']->[(0,'A'),(1,'B'),(2,'C')],
    # 这里是为了将索引传给step输出
    for step, (x, y) in enumerate(train_loader):
        output = cnn(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            # enumerate() 函数用于将一个可遍历的数据对象组合为一个索引序列。例:['A','B','C']->[(0,'A'),(1,'B'),(2,'C')]
            for (test_x, test_y) in test_loader:
                # print(test_y.size())
                # 在所有数据集上预测精度：
                # 预测结果 test_output.size() = [10000,10],其中每一列代表预测为每一个数的概率(softmax输出),而不是0或1
                test_output = cnn(test_x)
                # torch.max()则将预测结果转化对应的预测结果,即概率最大对应的数字:[10000,10]->[10000]
                pred_y = torch.max(test_output,1)[1].squeeze() # squeeze()默认是将a中所有为1的维度删掉
                # pred_size() = [10000]
                accuracy = sum(pred_y == test_y) / test_data.__len__()
                print('Eopch:',
                      epoch,
                      ' | train loss: %.6f' % loss.item(),
                      ' | test accracy:%.5f' % accuracy,
                      ' | step: %d' % step)
