import torch
import torch.nn as nn
from models_components import PixelNorm, LinearEqualized, EqualizedConv2d, GenBlock
import torch.nn.functional as F

class Generator(nn.Module):
  """
  Only created once.
  """

  def __init__(self, in_channels):
    super().__init__()
    latent_blocks = [
        PixelNorm(),
        LinearEqualized(in_features=in_channels, out_features=4*4*512)
    ]
    self.latent_blocks = nn.Sequential(*latent_blocks) #This block will be reshape in the forward

    conv_4 = nn.Sequential(
        EqualizedConv2d(in_channels=in_channels, out_channels=in_channels),
        nn.LeakyReLU(0.2),
        PixelNorm(),
        EqualizedConv2d(in_channels=in_channels, out_channels=in_channels),
        nn.LeakyReLU(0.2),
        PixelNorm())

    # The next 3 gen blocks do not shrink
    gen_blocks = nn.ModuleList([conv_4]) # 0: 4

    for i in range(1, 4): #1: 8, 2: 16, 3: 32
      gen_blocks.append(
          GenBlock(in_channels=in_channels, mid_channels=in_channels, shrink=False, name=f'conv_{i}')
      )
    # The next 5 shrink
    for i in range(4,9): # 4: 64, 5: 128, 6: 256, 7: 512, 8: 1024
      gen_blocks.append(
          GenBlock(in_channels=in_channels, mid_channels=in_channels, shrink=True, name=f'conv_{i}')
      )
      in_channels = in_channels//2
    self.gen_blocks = gen_blocks

    self.toRGB_new_path = EqualizedConv2d(in_channels=512, out_channels=3, kernel_size=1, padding=0, stride=1, name="conv_0") # When we use for the resolution 4x4

  def forward(self, x, idx, alpha=1):
    """
    idx: the index in the gen_blocks, start with 0 which is the resolution 8x8, because the 4x4 resolution is self.conv_4
    """
    out = self.latent_blocks(x)

    out = out.view(-1, 512, 4, 4)
    out = self.gen_blocks[0](out)
    #print(f'first out {out.shape}')
    if idx == 0:
      #print(f'out shape at {idx}: {out.shape}')
      return self.toRGB_new_path(out)

    elif idx == 1:

      out = self.gen_blocks[1](out, alpha=alpha, use_rgb=True)
      #print(f'out shape at {1}: {out.shape}')
      return out

    else:
      for i in range(1, idx): #Keep the last one for use_rbg
        out = self.gen_blocks[i](out, alpha, use_rgb=False)
        #print(f'out shape at {i}: {out.shape}')
      out = self.gen_blocks[idx](out, alpha, use_rgb=True)
      #print(f'out shape at {idx}: {out.shape}')
    return F.tanh(out) # The paper don't mention this, but we use it here.



