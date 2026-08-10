import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR'):
    layers = []
    for c in mode:
        if c == 'C':
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif c == 'B':
            layers.append(nn.BatchNorm2d(out_channels))
        elif c == 'R':
            layers.append(nn.ReLU(inplace=True))
        elif c == 'L':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif c == '2':
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias))
        elif c == '3':
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias))
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, mid_channels=64, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

def downsample_strideconv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=2, padding=0, bias=True)

def upsample_convtranspose(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, stride=2, kernel_size=2, padding=0, bias=True)
