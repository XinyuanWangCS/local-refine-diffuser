# reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d.py
import torch 
import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

class ConditionResNet(nn.Module):
    def __init__(
        self,
        input_size = 32,
        in_channels = 4,
        block_output_channels = (64, 128, 256, 512),
        class_num = 2,
        ):
        super().__init__()
        self.input_size = input_size
        self.class_num = class_num
        
        time_embed_dim = block_output_channels[0] * 4
        timestep_input_dim = block_output_channels[0]
        
        self.time_proj = Timesteps(block_output_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        
        self.conv1 = nn.Conv2d(in_channels, block_output_channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(block_output_channels[0], eps=1e-05, momentum=0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        self.layer1 = ResnetBlock2D(in_channels=block_output_channels[0], 
                                    out_channels=block_output_channels[0], 
                                    temb_channels=time_embed_dim, 
                                    time_embedding_norm = 'scale_shift',
                                    down=True)
        self.layer2 = ResnetBlock2D(in_channels=block_output_channels[0], 
                                    out_channels=block_output_channels[1], 
                                    temb_channels=time_embed_dim, 
                                    time_embedding_norm = 'scale_shift',
                                    down=True)
        self.layer3 = ResnetBlock2D(in_channels=block_output_channels[1], 
                                    out_channels=block_output_channels[2], 
                                    temb_channels=time_embed_dim, 
                                    time_embedding_norm = 'scale_shift',
                                    down=True)
        self.layer4 = ResnetBlock2D(in_channels=block_output_channels[2], 
                                    out_channels=block_output_channels[3], 
                                    temb_channels=time_embed_dim, 
                                    time_embedding_norm = 'scale_shift',
                                    down=True)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(block_output_channels[3], class_num)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, t=None):
        emb = None
        if t is not None:
            t_emb = self.time_proj(t)
            emb = self.time_embedding(t_emb)

        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x, temb=emb)
        x = self.layer2(x, temb=emb)
        x = self.layer3(x, temb=emb)
        x = self.layer4(x, temb=emb)
        x = self.avg_pool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
