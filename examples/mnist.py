import sys 
sys.path.append(r"D:\code\efficient_kan")
from src.efficient_kan import KAN

# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load MNIST
# 定义数据预处理转换：将图像转换为张量，并进行归一化
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
# 创建训练数据集
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
valset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
# 创建训练数据加载器
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# Define model
model = KAN([28 * 28, 64, 10]) # 输入特征为28*28，有一个隐藏层（64个神经元），输出层有10个神经元，用于手写数字分类任务
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 将模型移动到可用的设备上（GPU 或 CPU）
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# 定义学习率调度器
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# 定义损失函数 计算模型输出与实际标签之间的损失
criterion = nn.CrossEntropyLoss()
for epoch in range(10):
     # 训练阶段
    model.train()
    # 使用 tqdm 显示训练进度条
    with tqdm(trainloader) as pbar:
         # 遍历训练数据加载器中的每个批次
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28).to(device)  # 将图像数据展平为一维张量，并移到指定设备上
            optimizer.zero_grad() # 梯度清零
            output = model(images) # 前向传播计算输出
            loss = criterion(output, labels.to(device)) # 计算损失
            loss.backward() # 反向传播
            optimizer.step()  # 更新优化器参数
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean() # 计算当前批次的平均准确率
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])  # 在进度条中显示当前损失、准确率和学习率

    # 验证阶段
    model.eval()
    val_loss = 0
    val_accuracy = 0
    # 禁用梯度计算
    with torch.no_grad():
        # 遍历验证数据加载器中的每个批次
        for images, labels in valloader:
            # 将图像数据展平为一维张量，并移到指定设备上
            images = images.view(-1, 28 * 28).to(device)
            output = model(images) # 前向传播计算输出
            val_loss += criterion(output, labels.to(device)).item() # 计算验证损失
            val_accuracy += ( # 计算验证准确率
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader) # 计算平均验证损失和平均验证准确率
    val_accuracy /= len(valloader)

    # 更新学习率
    scheduler.step()
    # 打印当前 epoch 的验证损失和验证准确率
    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )
