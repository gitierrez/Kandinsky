import torch


def ConvolutionalBlock(in_channels, out_channels, kernel_size, stride, padding=0, norm=False, relu=False):
    layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if norm:
        layers.append(torch.nn.InstanceNorm2d(out_channels))
    if relu:
        layers.append(torch.nn.ReLU())
    return layers


class ResidualBlock(torch.nn.Module):
    
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.convs = torch.nn.Sequential(
            *ConvolutionalBlock(in_channels, in_channels, 3, 1, norm=True, relu=True),
            *ConvolutionalBlock(in_channels, in_channels, 3, 1, norm=True, relu=False)
        )
        self.downsample = torch.nn.Sequential(
            *ConvolutionalBlock(in_channels, in_channels, 5, 1, norm=False, relu=False)
        )
        
    def forward(self, x):
        return self.convs(x) + self.downsample(x)