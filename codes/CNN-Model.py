
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.inorm1 = nn.InstanceNorm3d(in_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.inorm2 = nn.InstanceNorm3d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.inorm1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.inorm2(out)
        out += residual  
        return out

class UpscalingBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, padding):
        super(UpscalingBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.prelu(self.conv(x))
        return out

class ResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, kernel_size, padding):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=kernel_size, padding=padding)
        self.prelu1 = nn.PReLU()
        self.resblock1 = ResidualBlock(base_channels, kernel_size, padding)
        self.upblock1 = UpscalingBlock(base_channels, kernel_size, padding)
        self.upblock2 = UpscalingBlock(base_channels, kernel_size, padding)
        self.conv2 = nn.Conv3d(base_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()  

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.resblock1(x)
        x = self.upblock1(x)
        x = self.upblock2(x)
        x = self.relu(self.conv2(x))
        return x
