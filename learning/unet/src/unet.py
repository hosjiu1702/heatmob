import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([Block(channels[idx], channels[idx+1]) for idx in range(len(channels) - 1)])
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        feats = []
        for block in self.encoder_blocks:
            x = block(x)
            feats.append(x)
            x = self.pool(x)
        return feats


if __name__ == '__main__':
    encoder = Encoder()
    x = torch.randn(1, 3, 572, 572)
    feats = encoder(x)
    for feat in feats:
        print(feat.shape)
