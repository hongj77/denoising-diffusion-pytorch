import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import torch
import pdb

class SinusoidalPositionEmbeddings(nn.Module):
  def __init__(self, dim):
      super().__init__()
      self.dim = dim

  def forward(self, time):
      device = time.device
      half_dim = self.dim // 2
      embeddings = math.log(10000) / (half_dim - 1)
      embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
      embeddings = time[:, None] * embeddings[None, :]
      embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
      return embeddings

class SelfAttention(nn.Module):
  def __init__(self, in_channels, key_dim):
    super().__init__()
    self.query_proj = nn.Conv2d(in_channels, key_dim, 1)
    self.key_proj = nn.Conv2d(in_channels, key_dim, 1)
    # Keep the same number of channels as the image.
    self.value_proj = nn.Conv2d(in_channels, in_channels, 1)
    self.key_dim = key_dim
    self.norm = nn.GroupNorm(32, in_channels)

  def forward(self, x):
    B, C, H, W = x.shape

    x = self.norm(x)

    # Shape: [B, N, key_dim]
    query = self.query_proj(x).reshape([B, H*W, -1])
    # Shape: [B, N, key_dim]
    key = self.key_proj(x).reshape([B, H*W, -1])
    # Shape: [B, N, N]
    attn = F.softmax((query @ key.permute(0,2,1)) * (self.key_dim)**-0.5, dim=-1)
    # Shape: [B, C, N]
    value = self.value_proj(x).reshape([B, -1, H*W])
    # Shape: [B, C, N]
    out = value @ attn
    out = einops.rearrange(out, 'b c (h w) -> b c h w', h=H, w=W)

    assert out.shape == x.shape
    return out + x

class ResnetBlock(nn.Module):
  """Resnet block with pre-activations from https://arxiv.org/pdf/1603.05027."""
  def __init__(self, in_channels, out_channels, time_channels, num_classes, num_groups=32, p_dropout=0.1):
    super().__init__()
    self.time_channels = time_channels
    self.out_channels = out_channels
    self.num_classes = num_classes

    self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
    self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
    self.act = nn.ReLU()
    self.dropout = nn.Dropout(p_dropout)

    # Preserves size.
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    # Preserves size.
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    # Zero initializing output conv layers can help learn the identity function better.
    nn.init.zeros_(self.conv2.weight)
    nn.init.zeros_(self.conv2.bias)
    # Expand channels for x to match the shape of h.
    self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    self.time_dense = nn.Linear(time_channels, out_channels)
    
    # I'm doing label embedding inside the resnet block unlike the time embedding
    # because the label would benefit from directly projecting to `out_channels`
    # which seems easier to do in this class vs. doing it in the unet.
    self.label_dense1 = nn.Linear(num_classes, out_channels)
    self.label_dense2 = nn.Linear(out_channels, out_channels)
    self.label_dense3 = nn.Linear(out_channels, out_channels)

  def forward(self, x, time_embed, label):
    B = x.shape[0]
    assert time_embed.shape == (B, self.time_channels)
    assert label.shape == (B,)

    # First convolution.
    h = self.act(self.norm1(x))
    h = self.conv1(h)

    # Add time as a condition.
    h += einops.rearrange(self.time_dense(self.act(time_embed)), 'b c -> b c 1 1')

    # Add label as a condition.
    label_one_hot = nn.functional.one_hot(label, self.num_classes).type(torch.float32)
    label_embed = self.label_dense1(label_one_hot)
    label_embed = self.label_dense2(self.act(label_embed))
    label_embed = self.label_dense3(self.act(label_embed))
    h += einops.rearrange(label_embed, 'b c -> b c 1 1')

    h = self.act(self.norm2(h))
    h = self.dropout(h)

    # Second convolution.
    h = self.conv2(h)
    x = self.conv_x(x)
    assert h.shape == x.shape

    # Output value is a real number. Will need activation function at the beginning of next block.
    return h + x 

class UpBlock(ResnetBlock):
  """Residual block where the returning shape is [B, C, H*2, W*2]"""
  def __init__(self, in_channels, out_channels, time_channels, num_classes, num_groups=32, p_dropout=0.1):
    super().__init__(in_channels, out_channels, time_channels, num_classes, num_groups, p_dropout)
    self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x, time_embed, label):
    B, C, H, W = x.shape
    h = super().forward(x, time_embed, label)
    # Double the output size.
    h = self.upsample(h)
    h = self.conv(h)
    assert h.shape == (B, self.out_channels, H*2, W*2)
    return h


class DownBlock(ResnetBlock):
  """Residual block where the returning shape is [B, C, H//2, W//2]"""
  def __init__(self, in_channels, out_channels, time_channels, num_classes, num_groups=32, p_dropout=0.1):
    super().__init__(in_channels, out_channels, time_channels, num_classes, num_groups, p_dropout)
    self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

  def forward(self, x, time_embed, label):
    B, C, H, W = x.shape
    h = super().forward(x, time_embed, label)
    h = self.downsample(h)
    assert h.shape == (B, self.out_channels, H//2, W//2)
    return h

class ConditionalUNet(nn.Module):
  """
  Original U-Net architecture: https://arxiv.org/pdf/1505.04597.
  Modified to add conditional timestep and label on each block.
  Attention is applied at the 16x16 resolution and at the bottleneck.
  """
  def __init__(self, num_classes=1, out_channels=3, time_embedding_dim=128, use_attn=True):
    super().__init__()
    self.num_classes = num_classes
    # 4 Down, 4 Up.
    self.channels = [64, 128, 256, 512, 1024]
    # Apply attention at the first downscaled resolution.
    self.attn_index = 1
    self.use_attn = use_attn
    # Convert C from 3 to 64. Keep same shape.
    self.conv1 = nn.Conv2d(out_channels, self.channels[0], kernel_size=3, stride=1, padding=1)
    # Convert C from 64 to 3. Keep same shape.
    self.conv2 = nn.Conv2d(self.channels[0], out_channels, kernel_size=3, stride=1, padding=1)
    # Zero initializing output conv layers can help learn the identity function better.
    nn.init.zeros_(self.conv2.weight)
    nn.init.zeros_(self.conv2.bias)

    self.act = nn.ReLU()
    self.norm = nn.GroupNorm(num_groups=32, num_channels=self.channels[0])

    self.time_embed = SinusoidalPositionEmbeddings(dim=time_embedding_dim)
    # Not sure why the time embedding dimension is 4x but the paper does this.
    time_channels = out_channels*4
    self.time_dense1 = nn.Linear(time_embedding_dim, time_channels)
    self.time_dense2 = nn.Linear(time_channels, time_channels)

    self.down_blocks = []
    for i in range(4):
      self.down_blocks.append(DownBlock(self.channels[i], self.channels[i+1], time_channels, num_classes))
    self.down_blocks = nn.ModuleList(self.down_blocks)

    self.up_blocks = []
    channels_reversed = list(reversed(self.channels))
    for i in range(4):
      # C is doubled because input is h + x.
      self.up_blocks.append(UpBlock(channels_reversed[i]*2, channels_reversed[i+1], time_channels, num_classes))
    self.up_blocks = nn.ModuleList(self.up_blocks)

    self.mid_block1 = ResnetBlock(self.channels[-1],self.channels[-1], time_channels, num_classes)
    self.mid_block2 = ResnetBlock(self.channels[-1],self.channels[-1], time_channels, num_classes)

    if use_attn:
      # Attention at the 16x16 resolution.
      self.down_attn = SelfAttention(self.channels[1], self.channels[1])
      # Attention at the 16x16 resolution.
      self.up_attn = SelfAttention(self.channels[1], self.channels[1])
      # Attention at the bottleneck resolution.
      self.mid_attn = SelfAttention(self.channels[-1], self.channels[-1])

  def forward(self, x, timestep, label):
    B, C, H, W = x.shape

    assert timestep.shape == (B,)
    assert label.shape == (B,)

    # C channels to 64 channels.
    h = self.conv1(x)

    # Get time embed.
    time_embed = self.time_dense1(self.time_embed(timestep))
    time_embed = self.time_dense2(self.act(time_embed))

    residuals = []
    for i, down_block in enumerate(self.down_blocks):
      h = down_block(h, time_embed=time_embed, label=label)
      if self.use_attn and i == self.attn_index-1:
        h = self.down_attn(h)
      residuals.append(h)
    
    # Apply attention at bottleneck layer.
    if self.use_attn:
      h = residuals[-1]
      h = self.mid_block1(h, time_embed=time_embed, label=label)
      h = self.mid_attn(h)
      h = self.mid_block2(h, time_embed=time_embed, label=label)

    for i, (up_block, residual) in enumerate(zip(self.up_blocks, reversed(residuals))):
      h = torch.cat([h, residual], dim=1)
      h = up_block(h, time_embed=time_embed, label=label)
      if self.use_attn and i == len(self.channels)-self.attn_index:
        h = self.up_attn(h)
    
    # Normalize and activate before conv2d.
    h = self.act(self.norm(h))
    return self.conv2(h)