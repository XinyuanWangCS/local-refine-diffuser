import torch.nn as nn
import torchvision.models as models

def extract_resnet_perceptual_outputs_v1(model, x):
    x = model(x)
    return [x]

def extract_resnet_perceptual_outputs_v1(model, x):
    layer_outputs = []

    # conv1
    x = model.resnet.conv1(x)
    x = model.resnet.bn1(x)
    x = model.resnet.relu(x)
    x = model.resnet.maxpool(x)
    layer_outputs.append(x)
    
    # layer1
    x = model.resnet.layer1(x)
    layer_outputs.append(x)

    # layer2
    x = model.resnet.layer2(x)
    layer_outputs.append(x)

    # layer3
    x = model.resnet.layer3(x)
    layer_outputs.append(x)

    # layer4
    x = model.resnet.layer4(x)
    layer_outputs.append(x)

    return layer_outputs

class ResNet(nn.Module):
    def __init__(self, resolution=32,  num_classes=1000):
        super(ResNet, self).__init__()
        self.resolution = resolution
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        if num_classes != 1000:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

class ResNet_v2(nn.Module):
    def __init__(self, resolution=32,  num_classes=1000):
        super(ResNet, self).__init__()
        self.resolution = resolution
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            # resnet.maxpool,  
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet_v1(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        if num_classes != 1000:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
    
class ResNet_v0(nn.Module):
    def __init__(self, resolution=32,  num_classes=1000):
        super(ResNet, self).__init__()
        self.resolution = resolution
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            # resnet.maxpool,  
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x