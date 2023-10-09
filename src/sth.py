import torch.nn as nn
import torch
from diffusers.models.resnet import ResnetBlock2D

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeEmbedding, self).__init__()
        self.embedding = nn.Embedding(embedding_dim)

    def forward(self, t):
        return self.embedding(t)

class ResNetDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial Conv Layer
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3)
        self.t_embedder = TimeEmbedding(128)
        
        # ResNet Blocks
        self.block1 = ResnetBlock2D(in_channels=64, out_channels=128, temb_channels=128, time_embedding_norm = 'scale_shift')
        self.block2 = ResnetBlock2D(in_channels=128, out_channels=256, temb_channels=256, time_embedding_norm = 'scale_shift')
        self.block3 = ResnetBlock2D(in_channels=256, out_channels=512, temb_channels=512, time_embedding_norm = 'scale_shift')
        
        # Pooling Layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer for Classification
        self.fc = nn.Linear(512, 1)
        
        # Sigmoid to convert to probability
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, t):
        x = self.conv1(x)
        t = self.t_embedder(t)
        
        x = self.block1(x, t)
        x = self.block2(x, t)
        x = self.block3(x, t)
        
        x = self.avg_pool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        x = self.sigmoid(x)
        
        return x

# Initialize the discriminator
discriminator = ResNetDiscriminator()

# Example input (considering you want to use 16 out of 64 in the batch due to torch.Size([16]) for time embedding)
# You'd generally match these in practice
x = torch.randn(16, 4, 32, 32)
t = torch.randn(16)

# Forward Pass
output = discriminator(x, t)
print(output)