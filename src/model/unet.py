import torch.nn as nn
import einops
import math
import torch

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

class ConditionalUNetBlock(nn.Module):
  """
  Conv (3x3) - preserve size
  BN
  Relu
  + <-------- t_embed, label
  Conv (3x3) - either scale up 2x or 1/2x
  BN
  RELU

  Conv 1 preserves the size. img features extracted. Other features concatted.
  Conv 2 either scales 2x or 1/2x
  """
  def __init__(self, in_channels, out_channels, time_embedding_dim, downsize=False, use_label=False, num_classes=None):
    super().__init__()

    self.time_embedding = SinusoidalPositionEmbeddings(dim=time_embedding_dim)
    self.time_mlp = nn.Linear(time_embedding_dim, out_channels)
    self.use_label = use_label
    self.num_classes = num_classes

    # Conv1 preserves the size.
    if downsize:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    else:
      # The residual is concatted, so the number of input channels is doubled.
      self.conv1 = nn.Conv2d(2*in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    self.relu = nn.ReLU()
    self.bn1 = nn.BatchNorm2d(num_features=out_channels)

    feature_channels = out_channels

    self.conv2 = nn.Conv2d(feature_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(num_features=out_channels)

    self.label_mlp = nn.Linear(num_classes, out_channels)

    # Scale the final output by either 2x or 0.5x.
    if downsize:
      self.final = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
    else:
      self.final = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

  def forward(self, x, t, label):
    """
      x - shape: [B, C, H, W]
      t - shape: [b,]
      label - shape: [b,]4
    """
    output = self.bn1(self.relu(self.conv1(x)))

    # Add timestep as a condition.
    time_embed = self.relu(self.time_mlp(self.time_embedding(t)))
    time_embed = einops.rearrange(time_embed, 'b c -> b c 1 1')
    # Time embedding is jsut added because it has the same channels?
    output = output + time_embed

    # Add label as a condition.
    if self.use_label:
      h, w = x.shape[-1], x.shape[-2]
      # If label is int, then we need to repeat it to fit the image shape.
      # Add the label as an additional channel.
      # Can I add one hot encoding?
      # MLP to turn label 1 channel into output_dims?
      label_one_hot = nn.functional.one_hot(label, self.num_classes).type(torch.float32)
      label_output = self.relu(self.label_mlp(label_one_hot))
      label_output = einops.repeat(label_output, 'b c -> b c h w', h=h, w=w)
      # MLP+Relu here? return b,c,h,w
      # They add it instead of catting the channels
      # output = torch.cat([output, label], dim=1)
      output = output + label_output

    output = self.bn2(self.relu(self.conv2(output)))
    return self.final(output)

class ConditionalUNet(nn.Module):
  """
  Original U-Net architecture: https://arxiv.org/pdf/1505.04597.
  Modified to add conditional timestep and label on each block.
  """
  def __init__(self, use_label=False, num_classes=None):
    super().__init__()
    # 4 Down, 4 Up
    self.channels = [64, 128, 256, 512, 1024]
    self.time_embedding_dim = 128
    self.use_label = use_label
    # Only valid if use_label is true.
    self.num_classes = num_classes

    self.down_blocks = []
    for i in range(4):
      self.down_blocks.append(ConditionalUNetBlock(self.channels[i], self.channels[i+1], self.time_embedding_dim, downsize=True, use_label=self.use_label, num_classes=num_classes))
    self.down_blocks = nn.ModuleList(self.down_blocks)

    self.up_blocks = []
    for i in range(4):
      channel_reversed = list(reversed(self.channels))
      self.up_blocks.append(ConditionalUNetBlock(channel_reversed[i], channel_reversed[i+1], self.time_embedding_dim, downsize=False, use_label=self.use_label, num_classes=num_classes))
    self.up_blocks = nn.ModuleList(self.up_blocks)

    image_channels = 3
    self.conv1 = nn.Conv2d(image_channels, self.channels[0], kernel_size=3, stride=1, padding=1)
    self.conv_output = nn.Conv2d(self.channels[0], image_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x, t, label=None):
    x = self.conv1(x)
    residuals = []
    for down_block in self.down_blocks:
      x = down_block(x, t, label)
      residuals.append(x)
    for up_block, residual in zip(self.up_blocks, reversed(residuals)):
      # Concat channels, double number of channels in the up block.
      x = torch.cat([x, residual], dim=1)
      x = up_block(x, t, label)
    return self.conv_output(x)
