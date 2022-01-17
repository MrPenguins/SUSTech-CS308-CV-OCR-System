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
import os
import matplotlib.pyplot as plt
#使用tensorboardX进行可视化
from tensorboardX import SummaryWriter

SumWriter = SummaryWriter(log_dir = "./EMNIST_log")
#print(torch.cuda.is_available())


EPOCH = 10 # FIXME
BATCH_SIZE = 128 # FIXME
LR = 1e-4 #FIXME


# 预处理 转为tensor 以及 标准化
transform = T.Compose([
     #转为灰度图像:
     T.Grayscale(num_output_channels=1),
     #将图片转换为Tensor,归一化至(0,1):
     T.ToTensor(),
     #比如原来的tensor是三个维度的，值在0到1之间，经过以下变换之后就到了-1到1区间
     #T.Normalize([0.5], [0.5])
])


#数据集要作为一个文件夹读入：

#读取训练集：
train_data = ImageFolder(root="./emnist-chars74k_datasets/Train_png", transform=transform)
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE, shuffle=True)

#读取测试集：
test_data = ImageFolder(root="./emnist-chars74k_datasets/Test_png", transform=transform)
#之所以要将test_data转换为loader是因为网络不支持原始的ImageFolder类数据，到时候直接使用批训练，便是tensor类。
#batch_size为全部10000张testdata，在全测试集上测试精度
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=test_data.__len__())
label_num = len(train_data.class_to_idx)


#数据可视化：
to_img = T.ToPILImage()
a=to_img(test_data[0][0]) #size=[1, 28, 28]
plt.imshow(a)
plt.axis('off')
plt.show()


# 图片的标签对应其在哪个文件夹下
#print(train_data.class_to_idx)#打印所有标签
#print(test_data.imgs)#打印所有图片对应的路径及标签



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
            nn.Linear(160,label_num),
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





cnn = CNN()
torch.save(cnn.state_dict(), 'EMNIST_CNN.pkl')

cnn.load_state_dict(torch.load('EMNIST_CNN.pkl'))
cnn.train()


print(cnn)
#定义优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
#定义损失函数
loss_func = nn.CrossEntropyLoss()
#根据EPOCH自动更新学习率，2次EPOCH学习率减少为原来的一半:
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma = 0.6, last_epoch = -1)

for epoch in range(EPOCH):
    #enumerate() 函数用于将一个可遍历的数据对象组合为一个索引序列。例:['A','B','C']->[(0,'A'),(1,'B'),(2,'C')],
    #这里是为了将索引传给step输出
    for step, (x, y) in enumerate(train_loader):
        output = cnn(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            #enumerate() 函数用于将一个可遍历的数据对象组合为一个索引序列。例:['A','B','C']->[(0,'A'),(1,'B'),(2,'C')]
            for (test_x, test_y) in test_loader:

                #print(test_y.size())
                #在所有数据集上预测精度：
                #预测结果 test_output.size() = [10000,10],其中每一列代表预测为每一个数的概率(softmax输出),而不是0或1
                test_output = cnn(test_x)
                #torch.max()则将预测结果转化对应的预测结果,即概率最大对应的数字:[10000,10]->[10000]
                pred_y = torch.max(test_output,1)[1].squeeze() #squeeze()默认是将a中所有为1的维度删掉
                #pred_size() = [10000]
                accuracy = sum(pred_y == test_y) / test_data.__len__()
                print('Eopch:', epoch, ' | train loss: %.6f' % loss.item(), ' | test accracy:%.5f' % accuracy,  ' | step: %d' % step)



                #为tensorboardX添加可视化日志：
                #1.添加训练集损失
                # SumWriter.add_scalar("train loss:",loss.item()/20, global_step = 20)

                # #计算测试集精度
                # test_output = cnn(test_x)
                # pred_y = torch.max(test_output,1)[1].squeeze()
                # accuracy = sum(pred_y == test_y) / test_data.__len__()
                # #2.添加测试集精度
                # SumWriter.add_scalar("test accuracy:",accuracy.item(),20)

                # #预处理当前batch：
                # b_x_im = utils.make_grid(x, nrow = 16)
                # #3.添加一个batch图像的可视化
                # SumWriter.add_image('train image sample:', b_x_im, 20)

                # #4.添加直方图可视化网络参数分布：
                # for name, param in cnn.named_parameters():
                #     SumWriter.add_histogram(name, param.data.numpy(), 20)



    #scheduler.step()

#仅保存训练好的参数
torch.save(cnn.state_dict(), 'EMNIST_CNN.pkl')
