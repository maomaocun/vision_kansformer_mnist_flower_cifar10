# 项目讲解链接：https://www.bilibili.com/video/BV1Nf4211753/?spm_id_from=333.999.0.0
import sys 
sys.path.append(r"D:\code\efficient_kan")
from src.efficient_kan import KAN
import  matplotlib.pyplot as plt
# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
 
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
 
        width = int(out_channel * (width_per_group / 64.)) * groups
 
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
 
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class ResNet(nn.Module):
 
    def __init__(self,
                 block,
                 blocks_num,
                 set_device: None,      
                 num_classes=1000,
                 include_top=True,      
                 include_top_kan = False, 
                 groups=1,
                 width_per_group=64):
        super().__init__()
        self.include_top = include_top
        self.include_top_kan = include_top_kan
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group
 
        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.include_top_kan:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # torch.Size([2, 512, 1, 1])
            # self.linear = nn.Linear(512,64* block.expansion)
            self.kan = KAN([512 * block.expansion,64,num_classes])
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
 
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
 
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
 
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
  
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        if self.include_top_kan:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            # x = self.kan(x)
        return x
 
 
def resnet34(set_device,num_classes=1000, include_top=True,include_top_kan=False):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], set_device=set_device,num_classes=num_classes, include_top=include_top,include_top_kan=include_top_kan)
 
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_class = 10

model= resnet34(set_device = device,num_classes=num_class,
                include_top=True,
                include_top_kan=False
                ).to(device)
# Define model
# model = KAN([28 * 28, 64, 10]) # 输入特征为28*28，有一个隐藏层（64个神经元），输出层有10个神经元，用于手写数字分类任务
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 将模型移动到可用的设备上（GPU 或 CPU）
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# 定义学习率调度器
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# 定义损失函数 计算模型输出与实际标签之间的损失
criterion = nn.CrossEntropyLoss()
# Lists to store accuracy and loss for plotting
val_accs = []  # 存储验证准确率

for epoch in range(10):
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    scheduler.step()

    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
    val_accs.append(val_accuracy)

# 绘制验证准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), val_accs, marker='o', linestyle='-', color='b')
plt.title('Validation Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.savefig('resnet_mnist_acc.png')  # 保存绘图
plt.show()