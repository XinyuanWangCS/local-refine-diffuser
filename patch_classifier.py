import torch
import torch.nn as nn

#################################################################################
#                       Patch Classifier                                        #
#################################################################################

class PatchClassifier(nn.Module):
    def __init__(self, in_channel=4, patch_size=4, hidden_size=768, depth = 1, activation=None, norm=None) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth

        self.patch_embed = nn.Conv2d(in_channels=in_channel, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size)
        #self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.conv_layers = nn.Sequential(
              *[nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1) for _ in range(depth)]
              )
        #self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.classifer = nn.Conv2d(in_channels=hidden_size, out_channels=1, kernel_size=1, stride=1)
        self.activation = nn.SiLU() if activation is None else activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.patch_embed(x)
        #x = self.norm1(x)
        x = self.activation(x)
        x = self.conv_layers(x)
        #x = self.norm2(x)
        x = self.activation(x)
        x = self.classifer(x)
        x = x.flatten(1)
        x = self.sigmoid(x)
        return x