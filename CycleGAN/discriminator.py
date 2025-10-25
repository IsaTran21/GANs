import torch
import torch.nn as nn
class Discriminator(nn.Module):
  """The CyCle GAN discriminator/critic.
    I use the WGAN-GP, then it should have been called critic, but anyway, here it is Discriminator (the initial version of the authors use it).
    This Discriminator does not have activation for its output.
  """

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv1 = self.build_conv_unit(in_channels=in_channels, out_channels=out_channels, use_norm=False) #Don't use norm
    self.conv2 = self.build_conv_unit(in_channels=out_channels, out_channels=out_channels*2) # Use norm
    self.conv3 = self.build_conv_unit(in_channels=out_channels*2, out_channels=out_channels*4) # Use norm
    self.conv4 = self.build_conv_unit(in_channels=out_channels*4, out_channels=out_channels*8, stride=1) # Use norm
    self.conv5 = self.build_conv_unit(in_channels=out_channels*8, out_channels=1, stride=1, use_norm=False, use_act=False) # Don't use norm, don't use activation


  def build_conv_unit(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_norm=True, use_act=True):
    """
    This methods create each conv unit block: conv2d -> instance norm (or not) -> leaky ReLU (slope=0.2) (or not)
    use_norm: True or False, if true then we will use instance norm and then bias in the previous conv2d = false.
    """

    act = nn.LeakyReLU(negative_slope=0.2) if use_act else nn.Identity()

    if use_norm:
      # When we use normalization, then bias = False
      conv_list = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.InstanceNorm2d(out_channels),
        act
    ]
    else:
      # When we don't use normalization, then bias = True
      conv_list = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        act
    ]

    return nn.Sequential(*conv_list)

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)

    return out