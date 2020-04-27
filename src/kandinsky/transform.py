import torch
from .layers import ConvolutionalBlock, ResidualBlock


class TransformNet(torch.nn.Module):
    
    def __init__(self):
        super(TransformNet, self).__init__()
        self.ref_pad = torch.nn.ReflectionPad2d(padding=40)
        self.convs = torch.nn.Sequential(
            *ConvolutionalBlock(3, 8, 9, 1, 4, norm=False, relu=False),
            *ConvolutionalBlock(8, 16, 3, 2, 1, norm=True, relu=True),
            *ConvolutionalBlock(16, 32, 3, 2, 1, norm=True, relu=True),
        )
        self.resblocks = torch.nn.Sequential(
            *[ResidualBlock(32) for i in range(5)]
        )
        self.deconvs = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            torch.nn.InstanceNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3, 2, 1, 1),
            torch.nn.InstanceNorm2d(8),
            torch.nn.ReLU()
        )
        self.final_conv = torch.nn.Sequential(
            *ConvolutionalBlock(8, 3, 9, 1, 4, norm=False, relu=False)
        )
    
    def forward(self, x):
        x = self.ref_pad(x)
        x = self.convs(x)
        x = self.resblocks(x)
        x = self.deconvs(x)
        x = self.final_conv(x)
        return x

    
if __name__ == '__main__':
    transform_net = TransformNet()
    sample_image = torch.randint(low=0, high=255, size=(1, 3, 256, 256))/255.
    assert transform_net(sample_image).shape == (1, 3, 256, 256)