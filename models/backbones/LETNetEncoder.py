import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class ConvMo(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output



class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
    
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = ConvMo(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.conv3x1 = ConvMo(nIn // 2, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.conv1x3 = ConvMo(nIn // 2, nIn // 2, (1, kSize), 1, padding=(0, 1), bn_acti=True)

        self.dconv3x1 = ConvMo(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = ConvMo(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ca11 = eca_layer(nIn // 2)

        self.ddconv3x1 = ConvMo(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = ConvMo(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ca22 = eca_layer(nIn // 2)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = ConvMo(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle = ShuffleBlock(nIn // 2)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)
        output = self.conv3x1(output)
        output = self.conv1x3(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.ca11(br1)

        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.ca22(br2)

        output = br1 + br2 + output
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        output = self.shuffle(output + input)

        return output

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = ConvMo(nIn, nConv, kSize=3, stride=2, padding=1)#
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output

class LETNetEncoder(nn.Module):
    DILATION_BLOCK = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32]
    def __init__(self, block_1=3, block_2=12, block_3=12, block_4=12):
        super().__init__()
        # Initial convolution layer
        self.init_conv = nn.Sequential(
            ConvMo(3, 32, 3, 1, padding=1, bn_acti=True),
            ConvMo(32, 32, 3, 1, padding=1, bn_acti=True),
            ConvMo(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.bn_prelu_1 = BNPReLU(32)

        # Downsampling block 1
        #self.downsample_1 = DownSamplingBlock(32, 64)

        # DAB Block 1
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(32, d=2))
        self.bn_prelu_2 = BNPReLU(32)

        # DAB Block 2
        self.downsample_2 = DownSamplingBlock(32, 64)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(64, d=self.DILATION_BLOCK[i]))
        self.bn_prelu_3 = BNPReLU(64)

        # DAB Block 3
        self.downsample_3 = DownSamplingBlock(64, 128)
        self.DAB_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.DAB_Block_3.add_module("DAB_Module_3_" + str(i),
                                        DABModule(128, d=self.DILATION_BLOCK[i]))
        self.bn_prelu_4 = BNPReLU(128)

        # DAB Block 4 (new addition)
        self.downsample_4 = DownSamplingBlock(128, 256)
        self.DAB_Block_4 = nn.Sequential()
        for i in range(0, block_4):
            self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
                                        DABModule(256, d=self.DILATION_BLOCK[i]))
        self.bn_prelu_5 = BNPReLU(256)

    def forward(self, input):
        # Initial convolution and activation
        output0 = self.init_conv(input)
        output0 = self.bn_prelu_1(output0)

        # DAB Block 1
        #output1_0 = self.downsample_1(output0)
        output1 = self.DAB_Block_1(output0)
        output1 = self.bn_prelu_2(output1)
        #print(f"DAB Block 1 (letnet编码器第一层的输出): {output1.shape}")

        # DAB Block 2
        output2_0 = self.downsample_2(output1)
        output2 = self.DAB_Block_2(output2_0)
        output2 = self.bn_prelu_3(output2)
        #print(f"DAB Block 2 (letnet编码器第二层的输出): {output2.shape}")

        # DAB Block 3
        output3_0 = self.downsample_3(output2)
        output3 = self.DAB_Block_3(output3_0)
        output3 = self.bn_prelu_4(output3)
        #print(f"DAB Block 3 (letnet编码器第三层的输出): {output3.shape}")

        # DAB Block 4 (new addition)
        output4_0 = self.downsample_4(output3)
        output4 = self.DAB_Block_4(output4_0)
        output4 = self.bn_prelu_5(output4)
        #print(f"DAB Block 4 (letnet编码器第四层的输出): {output4.shape}")

        return [output1, output2, output3, output4] 