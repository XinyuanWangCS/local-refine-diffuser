import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from .layers import SNConv2d, SNLinear, SNEmbedding, Attention

# BigGAN-deep: uses a different resblock and pattern

class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
               preactivation=True, activation=None, downsample=None,
               channel_ratio=4):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels // channel_ratio
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0)
    self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv4 = self.which_conv(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0)
                                 
    self.learnable_sc = True if (in_channels != out_channels) else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels - in_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.downsample:
      x = self.downsample(x)
    if self.learnable_sc:
      x = torch.cat([x, self.conv_sc(x)], 1)    
    return x
    
  def forward(self, x):
    # 1x1 bottleneck conv
    h = self.conv1(F.relu(x))
    # 3x3 convs
    h = self.conv2(self.activation(h))
    h = self.conv3(self.activation(h))
    # relu before downsample
    h = self.activation(h)
    # downsample
    if self.downsample:
      h = self.downsample(h)     
    # final 1x1 conv
    h = self.conv4(h)
    return h + self.shortcut(x)
    
# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64',ksize='333333', dilation='111111'):
  arch = {}
  arch[256] = {'in_channels' :  [item * ch for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[128] = {'in_channels' :  [item * ch for item in [1, 2, 4,  8, 16]],
               'out_channels' : [item * ch for item in [2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[64]  = {'in_channels' :  [item * ch for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}
  arch[32]  = {'in_channels' :  [item * ch for item in [4, 4, 4]],
               'out_channels' : [item * ch for item in [4, 4, 4]],
               'downsample' : [True, True, True, False],
               'resolution' : [16, 16, 16, 16],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,6)}}
  return arch

class BigGANClassifier(nn.Module):
  def __init__(self, in_channels=4, D_ch=64, D_wide=True, D_depth=2, resolution=32,
               D_kernel_size=3, D_attn='64', t_condition=None,#1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               SN_eps=1e-12, output_dim=2, 
               D_init='ortho', skip_init=False, D_param='SN', **kwargs):
    super(BigGANClassifier, self).__init__()
    # Width multiplier
    self.ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = D_wide
    # How many resblocks per stage?
    self.D_depth = D_depth
    # Resolution
    self.resolution = resolution
    # Kernel size
    self.kernel_size = D_kernel_size
    # Attention?
    self.attention = D_attn
    # Number of classes
    self.t_condition = t_condition
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init = D_init
    # Parameterization style
    self.D_param = D_param
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # Architecture
    self.arch = D_arch(self.ch, self.attention)[resolution]
    self.in_channels = in_channels
    
    # Which convs, batchnorms, and linear layers to use
    # No option to turn off SN in D right now
    if self.D_param == 'SN':
      self.which_conv = functools.partial(SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(SNLinear,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_embedding = functools.partial(SNEmbedding,
                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                              eps=self.SN_eps)
      
    self.encoder = BigGANEncoder(in_channels=self.in_channels, D_ch=D_ch, D_wide=D_wide, D_depth=D_depth, resolution=resolution,
               D_kernel_size=D_kernel_size, D_attn=D_attn, 
               num_D_SVs=num_D_SVs, num_D_SV_itrs=num_D_SV_itrs, D_activation=D_activation,
               SN_eps=SN_eps, 
               D_init=D_init, skip_init=skip_init, D_param=D_param)
    
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    hidden_size = self.resolution
    for down in self.arch['downsample']:
      if down:
        hidden_size /= 2
    
    self.linear = self.which_linear(self.arch['out_channels'][-1] * (int(hidden_size))**2, output_dim)
    # Embedding for projection discrimination
    if self.t_condition is not None:
      self.embed = self.which_embedding(self.t_condition, self.arch['out_channels'][-1])
      
  def forward(self, x, t=None):
    h = self.encoder(x=x, t=t)
    # Get initial class-unconditional output
    out = self.linear(h)
    # Get projection of final featureset onto class vectors and add to evidence
    if t is not None:
      out = out + torch.sum(self.embed(t) * h, 1, keepdim=True)
      
    return out
  
  
class BigGANEncoder(nn.Module):

  def __init__(self, in_channels=4, D_ch=64, D_wide=True, D_depth=2, resolution=32,
               D_kernel_size=3, D_attn='64', 
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               SN_eps=1e-12, D_init='ortho', skip_init=False, D_param='SN', **kwargs):
    super(BigGANEncoder, self).__init__()
    # Width multiplier
    self.ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = D_wide
    # How many resblocks per stage?
    self.D_depth = D_depth
    # Resolution
    self.resolution = resolution
    # Kernel size
    self.kernel_size = D_kernel_size
    # Attention?
    self.attention = D_attn
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init = D_init
    # Parameterization style
    self.D_param = D_param
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # Architecture
    self.arch = D_arch(self.ch, self.attention)[resolution]
    self.in_channels = in_channels


    # Which convs, batchnorms, and linear layers to use
    # No option to turn off SN in D right now
    if self.D_param == 'SN':
      self.which_conv = functools.partial(SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(SNLinear,
                          num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                          eps=self.SN_eps)
      self.which_embedding = functools.partial(SNEmbedding,
                              num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                              eps=self.SN_eps)
    
    # Prepare model
    # Stem convolution
    self.input_conv = self.which_conv(self.in_channels, self.arch['in_channels'][0])
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index] if d_index==0 else self.arch['out_channels'][index],
                        out_channels=self.arch['out_channels'][index],
                        which_conv=self.which_conv,
                        wide=self.D_wide,
                        activation=self.activation,
                        preactivation=True,
                        downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] and d_index==0 else None))
                        for d_index in range(self.D_depth)]]
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [Attention(self.arch['out_channels'][index],
                                              self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    self.flatten = nn.Flatten()

    # Initialize weights
    if not skip_init:
      self.init_weights()

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    #print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, t=None):
    # Run input conv
    h = self.input_conv(x)
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    
    #h = torch.sum(self.activation(h), [2, 3])
    return self.flatten(self.activation(h))

