
import typing as t
import torch
import torch.nn as nn
from einops import rearrange
from mmengine.model import BaseModule


class SCSA(BaseModule):
    def __init__(
            self,
            dims: t.List[int],  # 四个输入的通道数列表
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            norm_cfg: t.Dict = dict(type='BN'),
            act_cfg: t.Dict = dict(type='ReLU'),
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str ='sigmoid',
    ):
        super(SCSA, self).__init__()
        self.dims = dims
        self.head_num = head_num
        # print(f"dims: {dims}")
        # print(f"sum(dims): {sum(dims)}")
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        # 检查每个输入特征的通道数是否能被 4 整除
        for dim in dims:
            assert dim % 4 == 0, f'The dimension of input feature ({dim}) should be divisible by 4.'
        self.group_chans = [dim // 4 for dim in dims]

        self.local_dwc = nn.ModuleList([
            nn.Conv1d(chans, chans, kernel_size=group_kernel_sizes[0],
                      padding=group_kernel_sizes[0] // 2, groups=chans)
            for chans in self.group_chans
        ])
        self.global_dwc_s = nn.ModuleList([
            nn.Conv1d(chans, chans, kernel_size=group_kernel_sizes[1],
                      padding=group_kernel_sizes[1] // 2, groups=chans)
            for chans in self.group_chans
        ])
        self.global_dwc_m = nn.ModuleList([
            nn.Conv1d(chans, chans, kernel_size=group_kernel_sizes[2],
                      padding=group_kernel_sizes[2] // 2, groups=chans)
            for chans in self.group_chans
        ])
        self.global_dwc_l = nn.ModuleList([
            nn.Conv1d(chans, chans, kernel_size=group_kernel_sizes[3],
                      padding=group_kernel_sizes[3] // 2, groups=chans)
            for chans in self.group_chans
        ])
        self.sa_gate = nn.Softmax(dim=2) if gate_layer =='softmax' else nn.Sigmoid()
        self.norm_h = nn.ModuleList([nn.GroupNorm(4, dim) for dim in dims])
        self.norm_w = nn.ModuleList([nn.GroupNorm(4, dim) for dim in dims])

        self.conv_d = nn.Identity()
        self.norm = nn.ModuleList([nn.GroupNorm(1, dim) for dim in dims])
        self.q = nn.ModuleList([
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
            for dim in dims
        ])
        self.k = nn.ModuleList([
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
            for dim in dims
        ])
        self.v = nn.ModuleList([
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
            for dim in dims
        ])
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer =='softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode =='recombination':
                self.down_func = self.space_to_chans
                self.conv_d = nn.ModuleList([
                    nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
                    for dim in dims
                ])
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode =='max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def space_to_chans(self, x, idx):
        b, c, h, w = x.size()
        x = rearrange(x, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=self.window_size, p2=self.window_size)
        x = self.conv_d[idx](x)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x1, x2, x3, x4 的形状为 (B, C, H, W)
        """
        results = []
        inputs = [x1, x2, x3, x4]
        for idx, x in enumerate(inputs):
            # Spatial attention priority calculation
            b, c, h_, w_ = x.size()
            # (B, C, H)
            x_h = x.mean(dim=3)
            # print(f"x_h shape: {x_h.shape}")  # 打印 x_h 的形状进行调试
            # print(f"self.group_chans[{idx}]: {self.group_chans[idx]}")  # 打印 self.group_chans 的值进行调试
            l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans[idx], dim=1)
            # (B, C, W)
            x_w = x.mean(dim=2)
            l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans[idx], dim=1)

            x_h_attn = self.sa_gate(self.norm_h[idx](torch.cat((
                self.local_dwc[idx](l_x_h),
                self.global_dwc_s[idx](g_x_h_s),
                self.global_dwc_m[idx](g_x_h_m),
                self.global_dwc_l[idx](g_x_h_l),
            ), dim=1)))
            x_h_attn = x_h_attn.view(b, c, h_, 1)

            x_w_attn = self.sa_gate(self.norm_w[idx](torch.cat((
                self.local_dwc[idx](l_x_w),
                self.global_dwc_s[idx](g_x_w_s),
                self.global_dwc_m[idx](g_x_w_m),
                self.global_dwc_l[idx](g_x_w_l)
            ), dim=1)))
            x_w_attn = x_w_attn.view(b, c, 1, w_)

            x = x * x_h_attn * x_w_attn

            # Channel attention based on self attention
            # reduce calculations
            if isinstance(self.conv_d, nn.ModuleList):
                y = self.space_to_chans(x, idx)
            else:
                y = self.down_func(x)
                y = self.conv_d(y)
            _, _, h_, w_ = y.size()

            # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
            y = self.norm[idx](y)
            q = self.q[idx](y)
            k = self.k[idx](y)
            v = self.v[idx](y)
            # 重新计算 head_dim
            head_dim = c // self.head_num
            # (B, C, H, W) -> (B, head_num, head_dim, N)
            q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                          head_dim=int(head_dim))
            k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                          head_dim=int(head_dim))
            v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                          head_dim=int(head_dim))

            # (B, head_num, head_dim, head_dim)
            attn = q @ k.transpose(-2, -1) * (head_dim ** -0.5)
            attn = self.attn_drop(attn.softmax(dim=-1))
            # (B, head_num, head_dim, N)
            attn = attn @ v
            # (B, C, H_, W_)
            attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
            # (B, C, 1, 1)
            attn = attn.mean((2, 3), keepdim=True)
            attn = self.ca_gate(attn)

            x = attn * x
            results.append(x)

        return tuple(results)