import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import network
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader

EPOCH = 100
BATCH_SIZE = 64
LR = 0.0001


transform_train = transforms.Compose([
    transforms.ToTensor()])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])

# 加载数据集
# 5w 32 32 3
train_data = datasets.CIFAR10(root=os.getcwd(), train=True, transform=transform_train, download=True)
# 1w 32 32 3
test_data = datasets.CIFAR10(root=os.getcwd(), train=False, transform=transform_test, download=True)


# 数据分批

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# 使用自定义模型
model = network.Net()

# 定义损失函数，分类问题使用交叉信息熵，回归问题使用MSE
criterion = nn.CrossEntropyLoss()
# torch.optim来做算法优化,该函数甚至可以指定每一层的学习率，这里选用Adam来做优化器，还可以选其他的优化器
optimizer = optim.Adam(model.parameters(), lr=LR)
# optimizer = optim.SGD(model.parameters(), lr_0.0001=LR)
# 设置GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 模型和输入数据都需要to device
mode = model.to(device)

# 模型训练

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('cifar-10')
best_loss = 100
for epoch in range(EPOCH):
    loss_list = []
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        # 每100个batch记录一次loss
        if i % 100 == 0:
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + i)
    loss_epoch = sum(loss_list)/len(loss_list)
    print('epoch{} loss:{:.4f}'.format(epoch + 1, loss_epoch))
    # 保存效果最好的模型
    if loss_epoch < best_loss:
        best_loss = loss_epoch
        torch.save(model, 'cifar10_model.pt')

# 保存模型参数

# 模型加载
model = torch.load('cifar10_model.pt')
# 在测试集上进行测试
# model.eval()
model.train()

correct, total = 0, 0
for j, data in enumerate(test_loader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    # 前向传播
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total = total + labels.size(0)
    correct = correct + (predicted == labels).sum().item()
    # 准确率可视化
    if j % 20 == 0:
        writer.add_scalar("Test/Accuracy", 100.0 * correct / total, j)

print('准确率：{:.4f}%'.format(100.0 * correct / total))
















