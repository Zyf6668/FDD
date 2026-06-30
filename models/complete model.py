import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

from .backbones.letnet_encoder import LETNetEncoder
from .backbones.vssm_encoder import VSSMEncoder
from .modules.fusion import FusionBlock
from .modules.scsa import SCSA
from .modules.decoder import Decoder


class RS3Mamba(nn.Module):
    def __init__(
        self,
        decode_channels=32,
        dropout=0.1,
        window_size=8,
        num_classes=4,
    ):
        super().__init__()

        encoder_channels = [32, 64, 128, 256]
        ssm_dims = [32, 64, 128, 256]

        self.backbone = LETNetEncoder()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(32, eps=1e-5, affine=True),
        )

        self.vssm_encoder = VSSMEncoder(patch_size=2, in_chans=32)

        self.fuse = nn.ModuleList([
            FusionBlock(encoder_channels[i], ssm_dims[i])
            for i in range(4)
        ])

        self.middle = SCSA(
            dims=encoder_channels,
            head_num=4,
            window_size=7,
            group_kernel_sizes=[3, 5, 7, 9],
            down_sample_mode="avg_pool",
            gate_layer="sigmoid",
        )

        self.decoder = Decoder(
            encoder_channels=encoder_channels,
            decode_channels=decode_channels,
            dropout=dropout,
            window_size=window_size,
            num_classes=num_classes,
        )

    def wavelet_transform(self, x):
        coeffs = pywt.dwt2(x.detach().cpu().numpy(), "haar")
        LL, (LH, HL, HH) = coeffs

        LL = torch.as_tensor(LL, device=x.device, dtype=x.dtype)
        LH = torch.as_tensor(LH, device=x.device, dtype=x.dtype)
        HL = torch.as_tensor(HL, device=x.device, dtype=x.dtype)
        HH = torch.as_tensor(HH, device=x.device, dtype=x.dtype)

        return LL, LH, HL, HH

    def forward(self, x):
        h, w = x.shape[-2:]

        LL, LH, HL, HH = self.wavelet_transform(x)

        LL = F.interpolate(LL, size=(h, w), mode="bilinear", align_corners=False)
        HH = F.interpolate(HH, size=(h, w), mode="bilinear", align_corners=False)

        ssm_input = self.stem(LL)
        ssm_features = self.vssm_encoder(ssm_input)

        cnn_features = self.backbone(HH)

        fused_features = []
        for i in range(4):
            fused = self.fuse[i](cnn_features[i], ssm_features[i + 1])
            fused_features.append(fused)

        ty1, ty2, ty3, ty4 = self.middle(*fused_features)
        out = self.decoder(ty1, ty2, ty3, ty4, h, w)

        return out