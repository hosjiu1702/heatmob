import torch.nn as nn
from torch.nn import functional as F


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
    
    def forward(self, x):
        return F.relu(self._f + x)

    def _f(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x        


# ResNet-18
class ResNet18(nn.Module):
    def __init__(self, in_channels, out_channels=64, stride=2, padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=(7, 7), stride=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn1 = nn.LazyBatchNorm2d()
        self.residual_block1 = ResidualBlock()