import torch
import torch.nn as nn
from config import gen_block_large, gen_block_small
class ResNetBlock_Unit(nn.Module):
  """
  Each convolution unit in the resnet block, when we zoom in.
  (Image. 7 in the theory).
  In CycleGAN,t he ResNet input and output are the same.
  """
  def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1):
    super().__init__()
    self.resnet_block = self.build_res_unit_block(in_channels, out_channels, kernel_size, padding, stride)

  def build_res_unit_block(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int)->list:
    """
      This method build the basic convolution block, the image. 7 in the theory.
      Only use the ReflectionPad2d.
      Args:
      in_channels
      out_channels
      kernel_size
      padding
      use_bias
      Return the basic conv block (image. 7 in the theory).
    """
    # The conv with ReLU and no ReLU
    basic_conv = [
        # The first conv in image. 7
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(),

        # The second conv in image. 7
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.InstanceNorm2d(out_channels),

        ]
    # use_activation = nn.ReLU() if use_act else nn.Identity()
    #self.relu = nn.ReLU()

    return nn.Sequential(*basic_conv)

  def forward(self, x):
    """The forward method to create the resnet unit (image. 6 in theory)
      Add the input x (image) and the output of self.resnet_block, NOT concatenate them.
    """
    out = x + self.resnet_block(x)
    return out


class Generator(nn.Module):
  """The Cycle GAN generator, it returns one of the two generator networks based on CycleGAN"""

  def __init__(self, in_channels: int, out_channels: int=3, type_gen: str="large"):
    """The constructor of Generator_CycleGAN, initialized everytime an object of it created.
    Args:
    in_channels
    out_channels
    type_gen: "small" (6 Resnet blocks) or "large" (9 ResNet blocks)
    All the conv blocks in the downsample and upsample have this structure: conv2d -> Instance Norm -> ReLu.
    Except the last conv: using nn.Tanh()
    The arc_dict is the dict architecture of the 6 block res (small) or 9 block res (large) => for better management.
    The gen_block_small or gen_block_large is global variable
    """
    super().__init__()
    # First conv
    current_in = 64

    all_blocks = [
        nn.ReflectionPad2d(3),
        nn.Conv2d(in_channels, current_in, kernel_size=7, stride=1, padding=0),
        nn.InstanceNorm2d(current_in),
        nn.ReLU()
    ]


    # Downsample blocks: 2 upsample blocks
    current_out = 128
    for _ in range(2):
      # i start from 1 now
      all_blocks.extend([
          nn.Conv2d(current_in, current_out, kernel_size=3, stride=2, padding=1),
          nn.InstanceNorm2d(current_out),
          nn.ReLU()
      ])
      current_in = current_out
      current_out = current_out * 2

    # Out of the loop, i = 4 now
    # The ResNet Blocks
    # 6 for the small one and 9 for the large one, but, indeed, we don't care and does not need to loop through it
    # Because they all have the same last 3 conv blocks :>
    res_channels = 256
    if type_gen == "small":
        for _ in range (6): #Because i starts with 1
            all_blocks.append(ResNetBlock_Unit(in_channels=res_channels, out_channels=res_channels))
    else:
        for _ in range (9): #Because i starts with 1
            all_blocks.append(ResNetBlock_Unit(in_channels=res_channels, out_channels=res_channels))

    # Upsample blocks (only 2)
    up_channels = 128

    current_ins = res_channels
    current_outs = up_channels
    for _ in range(2):

        all_blocks.extend([
          nn.ConvTranspose2d(in_channels=current_ins, out_channels=current_outs, kernel_size=3, stride=2, padding=1, output_padding=1),
          nn.InstanceNorm2d(current_outs),
          nn.ReLU()
      ])
        current_ins = current_outs
        current_outs //= 2
    # The output conv
    all_blocks.extend([
        nn.ReflectionPad2d(3),
        nn.Conv2d(up_channels // 2, out_channels, kernel_size=7, stride=1, padding=0),
        nn.Tanh()
    ])

    self.model = nn.Sequential(*all_blocks)
    self._weights_init()

  def _weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

  def forward(self, x):
    return self.model(x)





if __name__ == "__main__":
    # Test the ResNetBlock
    x_test = torch.randn((10, 256, 128, 128))  # Number of samples, num channels, height, width
    res_net_block_test = ResNetBlock_Unit(256, 256)
    x_out = res_net_block_test(x_test)
    print(f'RESNET test {x_out.shape}')

    x_test = torch.randn((1, 3, 128, 128))
    gen_test_large = Generator(in_channels=3, type_gen="large")

    out_test_large = gen_test_large(x_test)
    print(out_test_large.shape)
    print(f' The min and max: {out_test_large.min()}, {out_test_large.max()}')