
from .base_model import BaseModel
import torch.nn as nn


class stage2Model(BaseModel):
    # 模型名称
    def name(self):
        return 'stage2Model'

    # 模型初始化 神经网络的初始化
    def initialize(self,label_num):
        self.Conv1 = nn.Sequential(
            # 卷积层1
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            # 激活函数层
            nn.ReLU(inplace=True),
            # 最大池化层
            nn.MaxPool2d(kernel_size=2)
        )
        self.Conv2 = nn.Sequential(
            # 卷积层2
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            # 激活函数层
            nn.ReLU(inplace=True),
            # 最大池化层
            nn.MaxPool2d(kernel_size=2)
        )
        # 最后接上一个全连接层(将图像变为1维)
        # 为什么是32*7*7：(1,28,28)->(16,28,28)(conv1)->(16,14,14)(pool1)->(32,14,14)(conv2)->(32,7,7)(pool2)->output
        self.Linear = nn.Sequential(
            nn.Linear(32 * 7 * 7, 800),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(800, 160),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(160, label_num),
        )

    def forward(self,input):
        input = self.Conv1(input)
        input = self.Conv2(input)  # view可理解为resize
        # input.size() = [100, 32, 7, 7], 100是每批次的数量，32是厚度，图片尺寸为7*7
        # 当某一维是-1时，会自动计算他的大小(原则是总数据量不变):
        input = input.view(input.size(0), -1)  # (batch=100, 1568), 最终效果便是将二维图片压缩为一维(数据量不变)
        # 最后接上一个全连接层，输出为10:[100,1568]*[1568,10]=[100,10]
        output = self.Linear(input)
        return output




