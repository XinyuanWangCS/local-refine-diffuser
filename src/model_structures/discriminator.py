import torch.nn as nn
import torchvision.models as models

class Pretrained_ResNet_Discriminator(nn.Module):
    def __init__(self, resnet, resolution=32, num_classes=2):
        super().__init__()  # Corrected this line
        self.resolution = resolution
        self.resnet = resnet
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

