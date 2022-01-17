import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
import torch


class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        self.transform = getTransform()


    def load_data(self):
        transform = self.transform
        train_data = ImageFolder(root=self.opt.train_data_path, transform=transform)
        test_data = ImageFolder(root=self.opt.test_data_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.opt.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=test_data.__len__())
        label_num = len(train_data.class_to_idx)

        return train_loader,test_loader,train_data,test_data,label_num

def getTransform():
    transform = T.Compose([
        # 转为灰度图像:
        T.Grayscale(num_output_channels=1),
        # 将图片转换为Tensor,归一化至(0,1):
        T.ToTensor(),
        # 比如原来的tensor是三个维度的，值在0到1之间，经过以下变换之后就到了-1到1区间
        # T.Normalize([0.5], [0.5])
    ])

    return  transform