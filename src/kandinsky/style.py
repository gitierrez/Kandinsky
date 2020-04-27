import torch
import torchvision


class StyleNet(torch.nn.Module):
    
    def __init__(self, base_model, output_layers):
        super(StyleNet, self).__init__()
        self.output_layers = output_layers
        self.features = base_model.features[:max(self.output_layers) + 1]
        self.freeze()
        
    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
            
    def forward(self, x):
        output = []
        for i, module in enumerate(list(self.features.modules())[1:]):
            x = module(x)
            if i in self.output_layers:
                output.append(x)
        return output


if __name__ == '__main__':
    sample_image = torch.randint(low=0, high=255, size=(1, 3, 512, 512))/255.
    stylenet = StyleNet(torchvision.models.vgg16(pretrained=True), output_layers=[3, 8, 15, 22])
    outputs = stylenet(sample_image)
    assert outputs[0].shape == (1, 64, 512, 512)
    assert outputs[1].shape == (1, 128, 256, 256)
    assert outputs[2].shape == (1, 256, 128, 128)
    assert outputs[3].shape == (1, 512, 64, 64)