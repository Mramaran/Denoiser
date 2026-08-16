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

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, mid_channels=64, out_channels=64):
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True)
        self.act = nn.GELU()
        
        self.norm2 = LayerNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=True)
        
    def forward(self, x):
        out = self.norm1(x)
        out = self.act(self.conv1(out))
        out = self.norm2(out)
        out = self.conv2(out)
        return x + out

def downsample_strideconv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=2, padding=0, bias=True)

def upsample_convtranspose(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
    )
