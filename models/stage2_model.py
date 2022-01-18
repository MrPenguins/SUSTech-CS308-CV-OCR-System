
from .base_model import BaseModel
import torch.nn as nn


# class stage2Model(BaseModel):
#     # 模型名称
#     def name(self):
#         return 'stage2Model'
#
#     def initialize(self,label_num):
#         self.Conv1 = nn.Sequential(
#             nn.Conv2d(1, 16, 5, 1, 2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.Conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 5, 1, 2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.Linear1 = nn.Sequential(
#             nn.Linear(32 * 7 * 7, 800),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#         )
#         self.Linear2 = nn.Sequential(
#             nn.Linear(800, 160),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True)
#         )
#         self.Linear3 = nn.Sequential(
#             nn.Linear(160, label_num)
#         )
#
#
#     def forward(self,input):
#         input = self.Conv1(input)
#         input = self.Conv2(input)
#         input = input.view(input.size(0), -1)
#         input = self.Linear1(input)
#         input = self.Linear2(input)
#         output = self.Linear3(input)
#
#         return output
#



class stage2Model(BaseModel):
    # 模型名称
    def name(self):
        return 'stage2Model'

    def initialize(self,label_num):
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
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
        input = self.Conv2(input)
        input = input.view(input.size(0), -1)
        output = self.Linear(input)
        return output


