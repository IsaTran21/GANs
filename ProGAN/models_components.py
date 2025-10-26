# Conv2dEqualized https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv2dEqualized https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
class EqualizedConv2d(nn.Module):
  """This class creates the custom conv2d which have the weights follow equlized learning rate"""

  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2, name=None):
      """
         The w is initialized from N(0, 1).
        b is initialized = 0
        gain = sqrt(gain/fan_in)
          fan_in = kernel_size * kernel_size * in_channels
      """
      super().__init__()
      self.name = name
      self.stride = stride
      self.padding = padding
      self.w = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size, kernel_size))) # Create the w, it is default that way
      fan_in = kernel_size * kernel_size * in_channels
      self.gain = torch.sqrt(torch.tensor(gain/fan_in))
      self.b = nn.Parameter(torch.zeros(out_channels,))

  def forward(self, x):
        """ """
        return F.conv2d(x, self.w * self.gain, self.b, stride=self.stride, padding=self.padding)

# Conv2dEqualized https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
class LinearEqualized(nn.Module):
  """This class creates the custom conv2d which have the weights follow equlized learning rate"""

  def __init__(self, in_features, out_features, gain=2, name=None):
    super().__init__()
    self.name = name
    self.w = nn.Parameter(torch.randn((out_features, in_features)))
    fan_in = torch.tensor(in_features)
    self.gain = 1 / torch.sqrt(fan_in)
    self.b = nn.Parameter(torch.zeros(out_features,))

  def forward(self, x):
    return F.linear(x, self.w * self.gain, self.b)
class PixelNorm(nn.Module):
    """
        Calculate the pixel normalization for each pixel across the channels.
        The x is of dim: (batch_size, channel, height, width), therefore, we just get the mean of x**2 acrros the first dim, the channels.
        After this, with the keepdim=True, we will have (batch_size, channel, height, width) / (batch_size, 1, height, width) => the channel will be broadcast.
        This only appears in the generator.
    """
    def __init__(self, epsilon=1e-12):
      super().__init__()
      self.epsilon = epsilon

    def forward(self, x):
      # The keepdim helps us like this: # (N, C, H, W) / (N, 1, H, W)
      return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class GenBlock(nn.Module):
  """
    The structure: Conv → LeakyReLU → PixelNorm
    Each GenBlock has 2 paths, the old_path, and the new_path.
    old_path: the images from the previous resolution will be upsized without going through the two convs
    new_path: the images fromt he previous resolution will be passed through the two convs and then down add with the old_path.
    As for the old_path, we only upsample it, and as for the new_path, the second conv output = the spatial dim of the output of the old_path.
    The idea: think of the old path as "left branch" and the new_path is the "right branch", then when we pass through a resolution, we “cut” all the left branches from earlier resolutions.
    All feature-producing convs (kernel size ≥ 3, including the first 4×4): Conv → LeakyReLU → PixelNorm
    Final toRGB conv (1×1): Conv → linear
  """
  def __init__(self, in_channels, mid_channels, kernel_size=3, padding=1, stride=1, name=None, shrink=False):
    """
    shrink is for the halves of the number of channels (because when the resolution increases, the num channels decrease for memory reason).
    It only happens when resolution >= 64.
    """
    super().__init__()

    # self.use_rgb = use_rgb

    if not shrink:
      out_channels = mid_channels
    else:
      out_channels = mid_channels // 2

    self.conv1 = EqualizedConv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, name=name)
    self.conv2 = EqualizedConv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, name=name)

    self.leaky = nn.LeakyReLU(0.2)
    self.pixel_norm = PixelNorm()
    blocks = [
          self.conv1,
          self.leaky,
          self.pixel_norm,
          self.conv2,
          self.leaky,
          self.pixel_norm
    ]
    # if self.use_rgb:
    self.toRGB_old_path = EqualizedConv2d(in_channels=in_channels, out_channels=3, kernel_size=1, padding=0, stride=1, name=name)
    self.toRGB_new_path = EqualizedConv2d(in_channels=out_channels, out_channels=3, kernel_size=1, padding=0, stride=1, name=name)

    self.blocks = nn.Sequential(*blocks)
    self.Tanh = nn.Tanh()

  def forward(self, x, alpha=1, use_rgb=False):
    # alpha goes from 0 → 1 gradually
    old_path = F.interpolate(x, scale_factor=2, mode='nearest') # (N, C, H, W) -> (N, C, 2H, 2W)
    new_path = self.blocks(old_path)

    if use_rgb:
      # We will use fade-in when we use rgb
      # Convert the old_path to rgb image
      old_path = self.toRGB_old_path(old_path) # (N, C, 2H, 2W) -> (N, 3, 2H, 2W)
      new_path = self.toRGB_new_path(new_path) # (N, C, 2H, 2W) -> (N, 3, 2H, 2W)
      out = (1 - alpha) * old_path + alpha * new_path # (N, 3, 2W, 2H)

    else:
      # if we don't use fade-in anymore, we will "cut the left branch"
      out = new_path # (N, C, W, H)

    return out
class MinibatchStd(nn.Module):
  """
  The minibatch std is for the progan implementation.
  The images are started from 4x4 resolution, therefore, the default value of expand_dim is 4.
  """

  def __init__(self, expand_dim=4, epsilon=1e-8):
    """
    Args:
    expand_dim: the spatial dim that the scalar will be expanded into.
    """
    super().__init__()
    self.expand_dim = expand_dim
    self.epsilon = epsilon

  def forward(self, x):
    """
    Args:
    x: (N, C, H, W)
    The scalar_std will be concatenated to the conv at resolution 4x4 in the discriminator
    """
    batch_size, channels, _, _ = x.shape
    # Calculate std across samples (N)
    std_sample = torch.std(x, dim=0) # (C, H, W)
    # print(f'std_sample.shape {std_sample.shape}')

    # Get the scalar value
    scalar_std = torch.mean(std_sample, dim=[0, 1, 2]) # (1,)

    # Expand into feature map
    #expanded_shape = (batch_size, 1, 4, 4) # In progan, we always use the minibatch std in the lowest dim, hence 4x4

    expanded_std = scalar_std.view(1, 1, 1, 1).expand(batch_size, 1, self.expand_dim, self.expand_dim)

    #expanded_std = torch.full(expanded_shape, scalar_std) # (N, 1, H, W)


    return expanded_std
class DiscBlock(nn.Module):
  """
  The downscale, the paper uses average pooling: nn.AvgPool2d((2,2))
  But, we can try: downscale = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

  """

  def __init__(self, in_channels, im_channel=3, kernel_size=3, padding=1, stride=1, name=None, expand=False):
    """
    When expand = True, then we have the output channels of the second conv (not including the conv 1x1) will be doubled.
    We need out_channels_old and out_channels_new because for the combination of old and new path. (Memory efficiency is the reason).
    Args:
    - out_channels_old: this is the output channels of the conv 1x1 (fromRGB of the old path). It does not always the same as the new path.
    - out_channels_new: this is the output channels of the conv 1x1 (fromRGB of the new path). It does not always the same as the old path.
    - in_channels: the in_channels of the first conv 3x3 in the 2 conv 3x3.
      + In the case of use_rbg: it is the output of the fromRGB block of the new_path = out_channels_new.
      + In the case of not use_rbg: it is the output of the previous layer. Example: in resolution 128x128, then the output of 256x256 resolution is the input of it.
    Vars:
    - mid_channels: the output channels of the first conv 3x3.
    - out_channels_conv: the output channels of the second conv 3x3.
    In case of expand = True, then out_channels_conv = mid_channels * 2, else: out_channels_conv = mid_channels.
    """
    super().__init__()
    self.leaky = nn.LeakyReLU(0.2)

    if expand:
      mid_channels = in_channels * 2

    else:
      mid_channels = in_channels

    out_channels_conv = mid_channels

    self.conv1 = EqualizedConv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, name=name)
    self.conv2 = EqualizedConv2d(in_channels=mid_channels, out_channels=out_channels_conv, kernel_size=kernel_size, padding=padding, stride=stride, name=name)

    # The two convs
    blocks = [
          # Conv 1
          self.conv1,
          self.leaky,
          # Conv 2
          self.conv2,
          self.leaky,
    ]
    self.blocks = nn.Sequential(*blocks)
    out_channels_old = out_channels_conv
    out_channels_new = in_channels
    self.fromRGB_old_path = EqualizedConv2d(in_channels=im_channel, out_channels=out_channels_old, kernel_size=1, padding=0, stride=1, name=name)
    self.fromRGB_new_path = EqualizedConv2d(in_channels=im_channel, out_channels=out_channels_new, kernel_size=1, padding=0, stride=1, name=name)
    self.downscale = nn.AvgPool2d((2,2))


  def forward(self, x, alpha=1, use_rgb=False):

      if use_rgb:
        old_path = self.downscale(x) # (N, C, H/2, W/2)

        old_path = self.fromRGB_old_path(old_path)


        new_path = self.fromRGB_new_path(x) # increase channels to out_channels_new

        new_path = self.blocks(new_path) # go through 2 new convs layers, output channels = out_channels_conv, if expand, then 2x, else, the same as input_channels

        new_path = self.downscale(new_path) # downscale the output of the 2 new conv layers


        out = (1 - alpha) * old_path + alpha * new_path
      else:
        out = self.blocks(x)
        out = self.downscale(out) # In the generator, we use the upscale for the old_path and then use it for the new_pathm
        # But here, when we don't use rgb, then we only need to downscale the new_path.
      return out