import torch
import torch.nn as nn
from models_components import LinearEqualized, EqualizedConv2d, DiscBlock, MinibatchStd


class Discriminator(nn.Module):

    def __init__(self, in_channels=64, im_channels=3):

        super().__init__()
        # The minibatch std for the lowest one

        self.im_channels = im_channels
        self.minibatch = MinibatchStd()
        self.linear_1 = LinearEqualized(in_features=512*4*4, out_features=512)
        self.linear_out = LinearEqualized(in_features=512, out_features=1)
        self.fromRGB_last = EqualizedConv2d(in_channels=im_channels, out_channels=512, kernel_size=1, padding=0, stride=1)

        # The lowest resolution
        conv_4 = nn.Sequential(
            # (512, 4, 4) → flatten to (8192) → linear(8192, 512) → linear(512, 1).
            # in_channels + 1 because of the minibatch std
            EqualizedConv2d(in_channels=512+1, out_channels=512),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(in_channels=512, out_channels=512),
            nn.LeakyReLU(0.2),
            ) # The output is being reshaped into into a vector of (512 * 4 * 4)
            # Then the output of if will be passed into self.linear (512*4*4, 512)
            # Then the output of it will be converted into 1
        self.conv_last = nn.Sequential(*conv_4)

        self.disc_blocks = nn.ModuleList([])

        # The expand layers: 1024 -> 512 -> 256 -> 128 -> 64
        for i in range(3): # (1024 -> 512 -> ) Only use:256 -> 128 -> 64: range(3)
            self.disc_blocks.append(
                DiscBlock(in_channels=in_channels, expand=True, name=f'conv_{i}')
            )
            in_channels = in_channels * 2

        # The un-expand layers:
        # now in_channels=512

        for i in range(3, 6): # 32 -> 16 -> 8: range(2, 5)
            self.disc_blocks.append(
                DiscBlock(in_channels=in_channels, expand=False, name=f'conv_{i}')
            )
        self.disc_blocks.append(self.conv_last)

    def forward(self, x, idx, alpha=1, use_rgb=False):
        """
        -1: 4x4; -2: 8x8; -3: 16x16; -4: 32x32; -5: 64x64; -6: 128x128;
        """

        minibatch = self.minibatch(x)
        # print(f'minibatch {minibatch.shape}')


        if idx == 0: # resolution 4x4
          # We still use from_RGB here, because the input of discriminator is always image of k channels.
          final_layer = self.fromRGB_last(x)
          final_layer = torch.concat((final_layer, minibatch), dim=1) # Along the channel: (N, 512+1, 4, 4)
          final_layer = self.disc_blocks[-1](final_layer) #512, 4, 4
          final_layer = final_layer.view(final_layer.shape[0], -1) # 512*4*4
          final_layer = self.linear_1(final_layer)
          final_layer = self.linear_out(final_layer) #(1, )
          return final_layer
        elif idx == 1: # resolution 8x8
          out = self.disc_blocks[-2](x, alpha=alpha, use_rgb=True) # Get the 8x8

        else:

          # The current one is use_rgb=True
          use_rbg_index = -(idx+1)
          out = self.disc_blocks[use_rbg_index](x, alpha=alpha, use_rgb=True) # Get the 8x8
          for i in range(idx, 1, -1): #6, 5, 4, 3, 2, don't count the 4x4 layer => exclude -1
            out = self.disc_blocks[-i](out, alpha=alpha, use_rgb=False) # Get the 8x8


        out = torch.concat((out, minibatch), dim=1) # Along the channel: (N, 512+1, 8, 8)
        out = self.disc_blocks[-1](out) #512, 4, 4
        out = out.view(x.shape[0], -1) # 512*4*4
        out = self.linear_1(out)
        out = self.linear_out(out) #(1, )
        return out




