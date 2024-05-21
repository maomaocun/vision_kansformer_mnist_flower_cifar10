import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# 项目讲解链接：https://www.bilibili.com/video/BV1Nf4211753/?spm_id_from=333.999.0.0
# 加载MNIST数据集并应用转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为PyTorch张量
    transforms.Normalize((0.5,), (0.5,))  # 将像素值归一化到范围[-1, 1]
])

# 加载训练和验证数据集
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
valset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 创建用于训练和验证的数据加载器
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)  # 全连接层，输入特征为28*28，输出特征为64
        self.fc2 = nn.Linear(64, 10)  # 全连接层，输入特征为64，输出特征为10（用于10分类任务）

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将输入张量展平为一维张量
        x = torch.relu(self.fc1(x))  # 对第一个全连接层的输出应用ReLU激活函数
        x = self.fc2(x)  # 应用第二个全连接层
        return x

# 创建MLP模型实例
model = MLP()

# 将模型移到可用设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 使用AdamW优化器，学习率为1e-3，权重衰减为1e-4
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 使用指数衰减的学习率调度器，衰减因子（gamma）为0.8
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(10):
    # 将模型设置为训练模式
    model.train()

    # 使用tqdm显示训练进度
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)  # 将图像移到指定设备
            labels = labels.to(device)  # 将标签移到指定设备

            optimizer.zero_grad()  # 梯度清零
            output = model(images)  # 前向传播
            loss = criterion(output, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            accuracy = (output.argmax(dim=1) == labels).float().mean()  # 计算准确率
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])  # 在进度条中显示损失、准确率和学习率

    # 验证
    model.eval()  # 将模型设置为评估模式
    val_loss = 0
    val_accuracy = 0

    with torch.no_grad():
        # 遍历验证数据集
        for images, labels in valloader:
            images = images.to(device)  # 将图像移到指定设备
            labels = labels.to(device)  # 将标签移到指定设备

            output = model(images)  # 前向传播
            val_loss += criterion(output, labels).item()  # 累积验证损失
            val_accuracy += ((output.argmax(dim=1) == labels).float().mean().item())  # 累积验证准确率

    val_loss /= len(valloader)  # 计算平均验证损失
    val_accuracy /= len(valloader)  # 计算平均验证准确率

    scheduler.step()  # 更新学习率

    # 打印当前epoch的验证损失和准确率
    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
