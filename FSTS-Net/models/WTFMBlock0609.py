import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from typing import Tuple, List, Optional


class DiscreteWaveletTransform2D(nn.Module):
    """
    二维离散小波变换(DWT)层 - 扩展支持Daubechies小波
    """

    def __init__(self, wavelet_type: str = 'haar'):
        super().__init__()
        self.wavelet_type = wavelet_type

        # 支持Haar和Daubechies小波滤波器
        if wavelet_type == 'haar':
            # 原有的Haar小波系数
            self.ll = torch.tensor([0.5, 0.5]).view(1, 1, 1, 2)
            self.lh = torch.tensor([-0.5, 0.5]).view(1, 1, 1, 2)
            self.hl = torch.tensor([0.5, 0.5]).view(1, 1, 2, 1)
            self.hh = torch.tensor([0.5, -0.5]).view(1, 1, 2, 1)

        elif wavelet_type == 'db2':
            # Daubechies 2小波系数
            lowpass = torch.tensor([0.4829629, 0.8365163, 0.2241439, -0.1294095])
            highpass = torch.tensor([-0.1294095, -0.2241439, 0.8365163, -0.4829629])

            self.ll = lowpass.view(1, 1, 1, 4)  # 行方向低通
            self.lh = highpass.view(1, 1, 1, 4)  # 行方向高通
            self.hl = lowpass.view(1, 1, 4, 1)  # 列方向低通
            self.hh = highpass.view(1, 1, 4, 1)  # 列方向高通

        elif wavelet_type == 'db4':
            # Daubechies 4小波系数（最常用）
            lowpass = torch.tensor([0.2303778, 0.7148466, 0.6308808, -0.0279838,
                                    -0.1870348, 0.0308414, 0.0328830, -0.0105974])
            highpass = torch.tensor([-0.0105974, -0.0328830, 0.0308414, 0.1870348,
                                     -0.0279838, -0.6308808, 0.7148466, -0.2303778])

            self.ll = lowpass.view(1, 1, 1, 8)
            self.lh = highpass.view(1, 1, 1, 8)
            self.hl = lowpass.view(1, 1, 8, 1)
            self.hh = highpass.view(1, 1, 8, 1)

        else:
            # 扩展错误信息，列出支持的小波类型
            supported_wavelets = ['haar', 'db2', 'db4']
            raise NotImplementedError(
                f"Wavelet type {wavelet_type} not implemented. "
                f"Supported types: {supported_wavelets}"
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对输入特征图进行二维小波变换 - 与原始WTFMBlock行为一致（无padding）
        """
        B, C, H, W = x.shape

        # 确保输入尺寸是2的倍数
        if H % 2 != 0 or W % 2 != 0:
            x = F.interpolate(x, size=(H - H % 2, W - W % 2), mode='bilinear', align_corners=False)
            B, C, H, W = x.shape

        # 扩展滤波器到正确的通道数
        ll = self.ll.expand(C, 1, 1, -1).to(x.device)
        lh = self.lh.expand(C, 1, 1, -1).to(x.device)
        hl = self.hl.expand(C, 1, -1, 1).to(x.device)
        hh = self.hh.expand(C, 1, -1, 1).to(x.device)

        # 在行方向进行卷积
        x_low_row = F.conv2d(x, ll, stride=(1, 2), padding=(0, 0), groups=C)
        x_high_row = F.conv2d(x, lh, stride=(1, 2), padding=(0, 0), groups=C)

        # 在列方向进行卷积
        cA = F.conv2d(x_low_row, hl, stride=(2, 1), padding=(0, 0), groups=C)
        cH = F.conv2d(x_low_row, hh, stride=(2, 1), padding=(0, 0), groups=C)
        cV = F.conv2d(x_high_row, hl, stride=(2, 1), padding=(0, 0), groups=C)
        cD = F.conv2d(x_high_row, hh, stride=(2, 1), padding=(0, 0), groups=C)

        return cA, cH, cV, cD


class WaveletTransformBlock(nn.Module):
    """
    小波变换块(WTB)
    对输入特征图进行小波变换并返回四个分量
    """

    def __init__(self, wavelet_type: str = 'haar'):
        super().__init__()
        self.dwt = DiscreteWaveletTransform2D(wavelet_type)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用小波变换

        参数:
            x: 输入特征图 [B, C, H, W]

        返回:
            cA, cH, cV, cD: 小波变换的四个分量
        """
        return self.dwt(x)


class WTFMBlock(nn.Module):
    """
    Wavelet Transform Feature Modulation (WTFM) 块
    通过多尺度频率表示增强浅层特征提取
    """

    def __init__(self, in_channels: int, out_channels: int, wavelet_type: str = 'haar'):
        """
        初始化WTFM块

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            wavelet_type: 小波类型
        """
        super().__init__()

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

        # 小波变换块
        self.wtb = WaveletTransformBlock(wavelet_type)

        # 第一个3x3卷积 (处理小波系数)
        self.conv_wavelet1 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1)
        # 第二个3x3卷积 (激活函数后)
        self.conv_wavelet2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 激活函数
        self.activation = nn.ReLU(inplace=True)

        # 上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 可选的批归一化
        self.bn3x3 = nn.BatchNorm2d(out_channels)
        self.bn7x7 = nn.BatchNorm2d(out_channels)
        self.bn_wavelet1 = nn.BatchNorm2d(out_channels)
        self.bn_wavelet2 = nn.BatchNorm2d(out_channels)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_ir: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x_ir: 输入特征图 [B, C, H, W]

        返回:
            f_combined: 综合特征图 [B, 3*out_channels, H, W]
        """
        B, C, H, W = x_ir.shape

        # 提取多尺度特征
        f = self.bn3x3(self.conv3x3(x_ir))  # 3x3卷积，提取细节特征
        f_prime = self.bn7x7(self.conv7x7(x_ir))  # 7x7卷积，提取全局特征

        # 对小尺度特征图f进行小波变换
        cA, cH, cV, cD = self.wtb(f)

        # 拼接小波分量
        f_wavelet = torch.cat([cA, cH, cV, cD], dim=1)  # 通道维度拼接

        # 第一个3x3卷积 + 激活函数
        wavelet_feat = self.activation(self.bn_wavelet1(self.conv_wavelet1(f_wavelet)))

        # 第二个3x3卷积
        wavelet_feat = self.bn_wavelet2(self.conv_wavelet2(wavelet_feat))

        # 上采样到原始尺寸
        wavelet_feat = self.upsample(wavelet_feat)

        # 确保尺寸匹配
        if wavelet_feat.size(2) != H or wavelet_feat.size(3) != W:
            wavelet_feat = F.interpolate(wavelet_feat, size=(H, W), mode='bilinear', align_corners=False)

        # 逐元素相乘调制
        f_double_prime = f_prime * wavelet_feat

        # 等式(12): 拼接所有特征
        f_combined = torch.cat([f, f_prime, f_double_prime], dim=1)

        return f_combined



# 测试代码
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = WTFMBlock(3,16).to(device)
    x = torch.randn(1, 3, 512, 512).to(device)
    out = model(x)
    print(out.shape)