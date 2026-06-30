from torch.optim import lr_scheduler
import copy
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import time
import cv2
import numpy as np

import torch.nn.init as init

# 参数初始化配置
trunc_normal_ = lambda x: init.trunc_normal_(x, mean=0., std=.02)
zeros_ = lambda x: init.constant_(x, 0)
ones_ = lambda x: init.constant_(x, 1)

# 转化成元组
def to_2tuple(x):
    return tuple([x] * 2)


# 独立层，nothing to da
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


# Patch Embedding 层
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_chans=256, embed_dim=1024):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# Attention 机制 （Multi-head Attention）
class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# MLP 多层感知机（本次就是全连接层）
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Drop_Path层
def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# Transformer Encoder
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, epsilon=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=epsilon)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim, eps=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# Visual Transformer (ViT)
class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_chans=256, class_dim=5, embed_dim=1024, depth=12,
                 num_heads=16, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, epsilon=1e-5, **args):
        super().__init__()
        self.class_dim = class_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_features = self.embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)  # 这里去掉了epsilon参数传入
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)  # 这里去掉了epsilon参数传入，采用默认值
        self.head = nn.Linear(embed_dim, class_dim) if class_dim > 0 else nn.Identity()

        trunc_normal_(self.pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        B = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        x = self.patch_embed(x)

        # 加入位置嵌入 Position Embedding
        x = x + self.pos_embed

        # Embedding Dropout
        x = self.pos_drop(x)

        # Transformer Encoder
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).reshape(B, C, H, W)
        x = torch.reshape(x, [B, 4 * C, H // 2, W // 2])
        return x

    def forward(self, x):
        y = self.forward_features(x)
        return y


# 第一层卷积块
class layer1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(layer1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y


# 第二层残差层: 每一层的第一个残差块中的3*3卷积执行下采样
# 跳连中带有下采样的
class Residual_block1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Residual_block1, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out)
        )
        self.relu = nn.ReLU(inplace=True)
        self.short = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        y = self.res(x)
        z = self.short(x)
        out = y + z
        out = self.relu(out)
        return out


# 跳连中不带下采样的
class Residual_block2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Residual_block2, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.res(x)
        out = y + x
        out = self.relu(out)
        return out


class Residual_block3(nn.Module):  # 用于第二层的第一个残差块，不改变分辨率但是调节通道数
    def __init__(self, ch_in, ch_out):
        super(Residual_block3, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out)
        )
        self.relu = nn.ReLU(inplace=True)
        self.short = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        y = self.res(x)
        z = self.short(x)
        out = y + z
        out = self.relu(out)
        return out


# 第二层
class layer2(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(layer2, self).__init__()
        self.block1 = Residual_block3(ch_in=channel_in, ch_out=channel_out)
        self.block2 = Residual_block2(ch_in=channel_out, ch_out=channel_out)
        self.block3 = Residual_block2(ch_in=channel_out, ch_out=channel_out)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.pool(x)
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        return y


# 第三层
class layer3(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(layer3, self).__init__()
        self.block1 = Residual_block1(ch_in=channel_in, ch_out=channel_out)
        self.block2 = Residual_block2(ch_in=channel_out, ch_out=channel_out)
        self.block3 = Residual_block2(ch_in=channel_out, ch_out=channel_out)
        self.block4 = Residual_block2(ch_in=channel_out, ch_out=channel_out)

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        return y


# 上采样卷积
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# TransUNet
class TransUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=4):
        super(TransUNet, self).__init__()

        self.firstlayer = layer1(ch_in=3, ch_out=64)
        self.secondlayer = layer2(channel_in=64, channel_out=128)
        self.thirdlayer = layer3(channel_in=128, channel_out=256)
        self.fourthlayer = VisionTransformer(ch_in=256, ch_out=1024)

        self.transconv = conv_block(ch_in=1024, ch_out=512)
        self.up1 = up_conv(ch_in=512, ch_out=256)
        self.conv1 = conv_block(ch_in=512, ch_out=256)
        self.up2 = up_conv(ch_in=256, ch_out=128)
        self.conv2 = conv_block(ch_in=256, ch_out=128)
        self.up3 = up_conv(ch_in=128, ch_out=64)
        self.conv3 = conv_block(ch_in=128, ch_out=64)
        self.up4 = up_conv(ch_in=64, ch_out=16)
        self.Conv_1x1 = nn.Conv2d(16, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 编码路径
        x1 = self.firstlayer(x)
        x2 = self.secondlayer(x1)
        x3 = self.thirdlayer(x2)
        x4 = self.fourthlayer(x3)
        x4=self.transconv(x4)
        
        # 解码端，注意在PyTorch中拼接操作使用torch.cat
        x4 = self.up1(x4)
        x3 = torch.cat([x4, x3], dim=1)
        x3 = self.conv1(x3)
        x3 = self.up2(x3)
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.conv2(x2)
        x2 = self.up3(x2)
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.conv3(x1)
        x1 = self.up4(x1)
        out = self.Conv_1x1(x1)

        return out
        