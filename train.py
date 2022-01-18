import torch
import torch.nn as nn

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model

# 获取超参数
opt = TrainOptions().parse()

# 数据载入阶段
data_loader = CreateDataLoader(opt)
train_loader,test_loader,train_data,test_data,label_num = data_loader.load_data()

# 模型初始化阶段
model = create_model(opt,label_num)

# 模型保存文件初始化
save_path = "./checkpoints/" + opt.name
torch.save(model.state_dict(), save_path)

model.load_state_dict(torch.load(save_path))

model.train()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)

# 损失函数
loss_func = nn.CrossEntropyLoss()


for epoch in range(opt.epoch):

    for step, (x, y) in enumerate(train_loader):
        output = model(x)
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
                test_output = model(test_x)
                #torch.max()则将预测结果转化对应的预测结果,即概率最大对应的数字:[10000,10]->[10000]
                pred_y = torch.max(test_output,1)[1].squeeze() #squeeze()默认是将a中所有为1的维度删掉
                #pred_size() = [10000]
                accuracy = sum(pred_y == test_y) / test_data.__len__()
                print('Eopch:', epoch, ' | train loss: %.6f' % loss.item(), ' | test accracy:%.5f' % accuracy,  ' | step: %d' % step)

    #仅保存训练好的参数
    torch.save(model.state_dict(), save_path+f'-epoch-{epoch}.pkl')







