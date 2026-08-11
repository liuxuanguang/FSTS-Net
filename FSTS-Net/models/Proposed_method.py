from models.Backbones.resnet import resnet18, resnet34, resnet50
from models.Decoders.Decoder import Seg_Decoder, CD_Decoder, Seg_Decoder_ResNet, CD_Decoder_ResNet
from models.Modules.CIEM import CIEM
from utils.misc import initialize_weights
from GrootV.classification.models.grootv import GrootVLayer, GrootV3DLayer, MTGrootV3DLayer
from GrootV.classification.models.grootv import GrootV, GrootV_3D
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pywt
import numpy as np
from typing import Tuple, List, Optional
from models.dual_vmamba import RGBXTransformer, vssm_tiny
import thop
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# class DiscreteWaveletTransform2D(nn.Module):
#     """
#     二维离散小波变换(DWT)层
#     将输入特征图分解为四个子带：cA, cH, cV, cD
#     """
#
#     def __init__(self, wavelet_type: str = 'haar'):
#         super().__init__()
#         self.wavelet_type = wavelet_type
#
#         # 使用Haar小波滤波器
#         if wavelet_type == 'haar':
#             # 低通和高通滤波器系数
#             self.ll = torch.tensor([0.5, 0.5]).view(1, 1, 1, 2)
#             self.lh = torch.tensor([-0.5, 0.5]).view(1, 1, 1, 2)
#             self.hl = torch.tensor([0.5, 0.5]).view(1, 1, 2, 1)
#             self.hh = torch.tensor([0.5, -0.5]).view(1, 1, 2, 1)
#         else:
#             # 可以扩展其他小波类型
#             raise NotImplementedError(f"Wavelet type {wavelet_type} not implemented")
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         对输入特征图进行二维小波变换
#
#         参数:
#             x: 输入特征图 [B, C, H, W]
#
#         返回:
#             cA: 近似系数 [B, C, H/2, W/2]
#             cH: 水平细节系数 [B, C, H/2, W/2]
#             cV: 垂直细节系数 [B, C, H/2, W/2]
#             cD: 对角线细节系数 [B, C, H/2, W/2]
#         """
#         B, C, H, W = x.shape
#
#         # 确保输入尺寸是2的倍数
#         if H % 2 != 0 or W % 2 != 0:
#             x = F.interpolate(x, size=(H - H % 2, W - W % 2), mode='bilinear', align_corners=False)
#             B, C, H, W = x.shape
#
#         # 将滤波器扩展到正确的通道数
#         ll = self.ll.expand(C, 1, 1, 2).to(x.device)
#         lh = self.lh.expand(C, 1, 1, 2).to(x.device)
#         hl = self.hl.expand(C, 1, 2, 1).to(x.device)
#         hh = self.hh.expand(C, 1, 2, 1).to(x.device)
#
#         # 在行方向进行卷积
#         x_low_row = F.conv2d(x, ll, stride=(1, 2), padding=(0, 0), groups=C)
#         x_high_row = F.conv2d(x, lh, stride=(1, 2), padding=(0, 0), groups=C)
#
#         # 在列方向进行卷积
#         cA = F.conv2d(x_low_row, hl, stride=(2, 1), padding=(0, 0), groups=C)
#         cH = F.conv2d(x_low_row, hh, stride=(2, 1), padding=(0, 0), groups=C)
#         cV = F.conv2d(x_high_row, hl, stride=(2, 1), padding=(0, 0), groups=C)
#         cD = F.conv2d(x_high_row, hh, stride=(2, 1), padding=(0, 0), groups=C)
#
#         return cA, cH, cV, cD


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


def get_backbone(backbone, pretrained):
    if backbone == 'resnet18':
        backbone = resnet18(pretrained)
    elif backbone == 'resnet34':
        backbone = resnet34(pretrained)
    elif backbone == 'resnet50':
        backbone = resnet50(pretrained)
    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)
    return backbone


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3_dw(in_channel,out_channel,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=stride,padding=1,groups=in_channel,bias=True),
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3_dw(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_dw(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResBlock1, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample

    def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


class STM_GrootV3D_V2(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_GrootV3D_V2, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        # self.conv2 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=128)
        self.GrootV_S1 = GrootV3DLayer(channels=self.inchannel)
        # self.smooth_layer_x = ResBlock1(in_channels=128, out_channels=128, stride=1)
        # self.smooth_layer_y = ResBlock1(in_channels=128, out_channels=128, stride=1)
    def forward(self, x, y):
        B, C, H, W = x.size()
        ct_tensor_42 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_42[:, :, :, 0:W] = x
        ct_tensor_42[:, :, :, W:2*W] = y
        # ct_tensor_42 = self.conv2(ct_tensor_42)
        if not self.channel_first:
            ct_tensor_42 = ct_tensor_42.permute(0, 2, 3, 1)
        f2 = self.GrootV_S1(ct_tensor_42)
        f2 = f2.permute(0, 3, 1, 2)

        xf_sm = f2[:, :, :, 0:W]
        yf_sm = f2[:, :, :, W:2*W]

        # xf_sm = self.smooth_layer_x(xf_sm)
        # yf_sm = self.smooth_layer_x(yf_sm)

        return xf_sm, yf_sm


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class DynamicGCN(nn.Module):
    def __init__(self, num_state):
        super().__init__()
        self.num_state = num_state
        self.key = nn.Conv1d(num_state, num_state, 1)
        self.query = nn.Conv1d(num_state, num_state, 1)
        self.value = nn.Conv1d(num_state, num_state, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        # 展平并检查维度
        x_flat = x.contiguous().view(B, C, N)
        assert C == self.num_state, f"通道数应为 {self.num_state}, 实际为 {C}"
        # 动态邻接矩阵
        k = self.key(x_flat).transpose(1, 2)  # [B, N, C]
        q = self.query(x_flat)  # [B, C, N]
        adj = torch.softmax(torch.bmm(k, q), dim=-1)  # [B, N, N]
        # 邻接聚合
        v = self.value(x_flat)  # [B, C, N]
        out = torch.bmm(adj, v.transpose(1, 2)).transpose(1, 2)  # [B, C, N]
        return out.contiguous().view(B, C, H, W)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=4, stride=4)
        in_channels = in_channels_high + in_channels_low//scale_ratio
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low//scale_ratio),
            nn.BatchNorm2d(in_channels_low//scale_ratio),
            nn.ReLU(inplace=True) )
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) )

    def forward(self, x, low_feat):
        x = self.up(x)
        low_feat = self.transit(low_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x


# 采用小波变化增强浅层特征,baseline
class mmscd_siam_GCN_WT(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_WT, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.WT_opt = DiscreteWaveletTransform2D(3, 16, 'db4')
        self.WT_sar = DiscreteWaveletTransform2D(3, 16)
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.GCN1 = GCN(80)
        self.GCN2 = GCN(160)
        self.GCN3 = GCN(320)
        self.GCN4 = GCN(640)
        self.GCN5 = GCN(640)
        self.GCN = [self.GCN1, self.GCN2, self.GCN3, self.GCN4, self.GCN5]
        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)

        else:
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]

        self.MambaLayer = STM_GrootV3D_V2(640, False)
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        # 加载预训练权重（关键修改）
        if pretrained:
            self.load_adapted_pretrained_weights()

        # 冻结backbone部分参数（可选）
        for param in self.backbone.parameters():
            param.requires_grad = True

        initialize_weights(self.CD_Decoder, self.classifierCD)

    def load_adapted_pretrained_weights(self):
        """智能加载适配的预训练权重"""
        try:
            # 加载预训练权重
            pretrained_weights = torch.load(
                '/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
            new_dict = pretrained_weights['model']

            # 创建适配的权重字典
            updated_weights = {}
            missing_keys = []
            unexpected_keys = []

            # 获取当前模型状态字典
            model_dict = self.backbone.state_dict()

            for key, value in new_dict.items():
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        # 形状完全匹配，直接加载
                        updated_weights[key] = value
                    elif key == 'patch_embed.conv1.weight':
                        # 特殊处理输入卷积层权重适配
                        adapted_weight = self.adapt_input_conv_weights(value, model_dict[key].shape)
                        updated_weights[key] = adapted_weight
                        print(f"✅ 适配输入层权重: {value.shape} -> {adapted_weight.shape}")
                    else:
                        # 其他不匹配的层，跳过并记录
                        missing_keys.append(key)
                        print(f"⚠️ 跳过不匹配的层: {key} {value.shape} -> {model_dict[key].shape}")
                else:
                    unexpected_keys.append(key)

            # 加载适配后的权重
            self.backbone.load_state_dict(updated_weights, strict=False)

            # 打印加载结果
            print("🎯 预训练权重加载完成!")
            print(f"📊 成功加载: {len(updated_weights)}/{len(model_dict)} 层")
            if missing_keys:
                print(f"⚠️ 跳过的层: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"❌ 未使用的预训练层: {len(unexpected_keys)} 个")

        except Exception as e:
            print(f"❌ 预训练权重加载失败: {e}")

    def adapt_input_conv_weights(self, original_weight, target_shape):
        """
        适配输入卷积层权重
        将3通道权重扩展到32通道
        """
        out_channels, in_channels, kh, kw = target_shape
        new_weight = torch.zeros(target_shape)

        # 计算重复因子
        repeat_factor = in_channels // original_weight.size(1)

        if repeat_factor > 0:
            # 将原始3通道权重复制到新的输入通道
            for i in range(repeat_factor):
                start_ch = i * original_weight.size(1)
                end_ch = (i + 1) * original_weight.size(1)
                if end_ch <= in_channels:
                    # 平均分配权重，保持数值稳定性
                    new_weight[:, start_ch:end_ch] = original_weight / repeat_factor

            # 处理剩余的通道（如果有）
            remainder = in_channels % original_weight.size(1)
            if remainder > 0:
                start_ch = repeat_factor * original_weight.size(1)
                new_weight[:, start_ch:start_ch + remainder] = original_weight[:, :remainder] / (repeat_factor + 1)
        else:
            # 如果目标通道数小于原始通道数，取前n个通道
            new_weight = original_weight[:, :in_channels] * (in_channels / original_weight.size(1))

        return new_weight

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def bi_mamba_forward(self, x, y):
        xf_sm1, yf_sm1 = self.MambaLayer(x, y)
        yf_sm2, xf_sm2 = self.MambaLayer(y, x)
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        return x_f, y_f


    def forward(self, opt, sar):
        opt = self.WT_opt(opt)
        sar = self.WT_sar(sar)
        b, c, h, w = opt.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = opt
        xy_in[:, :, :, w:2*w] = sar
        feature_xy = self.backbone.forward(xy_in)

        # 遍历A中的每个矩阵
        feature1 = []
        feature2 = []

        for matrix in feature_xy:
            # 在W维度上划分
            Bs, Cs, Hs, Ws = matrix.shape
            Ws = Ws//2
            T1_part = matrix[:, :, :, 0:Ws]  # 左半部分
            T2_part = matrix[:, :, :, Ws:2*Ws]  # 右半部分
            # 将各部分分别存储到feature列表中
            feature1.append(T1_part)
            feature2.append(T2_part)

        feature1_4, feature2_4 = self.bi_mamba_forward(feature1[-1], feature2[-1])
        feature1[-1] = feature1_4
        feature2[-1] = feature2_4

        # 遍历所有层级的特征
        for i in range(len(feature1)):
            # 获取当前层级的特征
            f1 = feature1[i]
            f2 = feature2[i]

            # 动态获取当前层级维度信息
            batch_size, channels, height, width = f1.size()

            gcn_f1 = f1.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f1 = self.GCN[i](gcn_f1)
            maps_f1 = map_f1.contiguous().view(batch_size, self.channel_nums[i], *f1.size()[2:])

            gcn_f2 = f2.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f2 = self.GCN[i](gcn_f2)
            maps_f2 = map_f2.contiguous().view(batch_size, self.channel_nums[i], *f2.size()[2:])

            restored_f1 = f1 + maps_f1
            restored_f2 = f2 + maps_f2
            # 更新特征列表
            feature1[i] = restored_f1
            feature2[i] = restored_f2

        # CDDecoder
        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature1[i], feature2[i]))

        xc = self.CD_Decoder(feature_diff)
        bcd = self.seg(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifierCD(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.softmax(scd)
        bcd = torch.sigmoid(bcd)
        return scd.squeeze(1), bcd.squeeze(1)


# add private and comment features
class AdaptiveFusion(nn.Module):
    def __init__(self, channels):
        """
        自适应特征融合模块
        :param channels: 输入特征图的通道数
        """
        super(AdaptiveFusion, self).__init__()
        self.channels = channels

        # 空间注意力机制
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 通道注意力机制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 融合权重生成网络
        self.fusion_weight = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, base_feat, gcn_feat):
        """
        :param base_feat: 原始特征图 [B, C, H, W]
        :param gcn_feat: GCN处理后的特征图 [B, C, H, W]
        :return: 融合后的特征图 [B, C, H, W]
        """
        # 空间注意力 - 识别重要空间区域
        # 组合平均池化和最大池化特征
        base_avg = torch.mean(base_feat, dim=1, keepdim=True)
        base_max, _ = torch.max(base_feat, dim=1, keepdim=True)
        gcn_avg = torch.mean(gcn_feat, dim=1, keepdim=True)
        gcn_max, _ = torch.max(gcn_feat, dim=1, keepdim=True)

        spatial_att_base = self.spatial_att(torch.cat([base_avg, base_max], dim=1))
        spatial_att_gcn = self.spatial_att(torch.cat([gcn_avg, gcn_max], dim=1))

        # 通道注意力 - 强化重要通道
        channel_att_base = self.channel_att(base_feat)
        channel_att_gcn = self.channel_att(gcn_feat)

        # 应用空间和通道注意力
        att_base_feat = base_feat * spatial_att_base * channel_att_base
        att_gcn_feat = gcn_feat * spatial_att_gcn * channel_att_gcn

        # 自适应融合权重生成
        combined = torch.cat([att_base_feat, att_gcn_feat], dim=1)
        fusion_weights = self.fusion_weight(combined)

        # 自适应融合
        fused_feat = fusion_weights * att_base_feat + (1 - fusion_weights) * att_gcn_feat

        # 残差连接保留原始信息
        fused_feat = base_feat + fused_feat

        return fused_feat


class AdaptiveFusion1(nn.Module):
    def __init__(self, channels):
        """
        自适应特征融合模块
        :param channels: 输入特征图的通道数
        """
        super(AdaptiveFusion1, self).__init__()
        self.channels = channels

        # 空间注意力机制
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 通道注意力机制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, channels // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 融合权重生成网络
        self.fusion_weight = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, base_feat, gcn_feat):
        """
        :param base_feat: 原始特征图 [B, C, H, W]
        :param gcn_feat: GCN处理后的特征图 [B, C, H, W]
        :return: 融合后的特征图 [B, C, H, W]
        """
        # 空间注意力 - 识别重要空间区域
        # 组合平均池化和最大池化特征
        base_avg = torch.mean(base_feat, dim=1, keepdim=True)
        base_max, _ = torch.max(base_feat, dim=1, keepdim=True)
        gcn_avg = torch.mean(gcn_feat, dim=1, keepdim=True)
        gcn_max, _ = torch.max(gcn_feat, dim=1, keepdim=True)

        spatial_att_base = self.spatial_att(torch.cat([base_avg, base_max], dim=1))
        spatial_att_gcn = self.spatial_att(torch.cat([gcn_avg, gcn_max], dim=1))

        # 通道注意力 - 强化重要通道
        channel_att_base = self.channel_att(base_avg + base_max)
        channel_att_gcn = self.channel_att(gcn_avg + gcn_max)

        # 应用空间和通道注意力
        att_base_feat = base_feat * spatial_att_base * channel_att_base
        att_gcn_feat = gcn_feat * spatial_att_gcn * channel_att_gcn

        # 自适应融合权重生成
        combined = torch.cat([att_base_feat, att_gcn_feat], dim=1)
        fusion_weights = self.fusion_weight(combined)

        # 自适应融合
        fused_feat = fusion_weights * att_base_feat + (1 - fusion_weights) * att_gcn_feat

        # 残差连接保留原始信息
        fused_feat = base_feat + fused_feat

        return fused_feat

class GCNWithAttention(nn.Module):
    """
    带有残差连接的注意力GCN
    最简单实用的版本
    """
    def __init__(self, num_state, num_node, bias=False):
        super(GCNWithAttention, self).__init__()
        self.num_state = num_state
        self.num_node = num_node

        # 图卷积
        self.gcn_conv = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv1d(num_state, num_state, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        # 残差连接
        residual = x

        # 图卷积
        h = self.gcn_conv(x)
        h = self.relu(h)

        # 注意力权重
        attn_weights = self.attention(h)

        # 应用注意力
        h_attended = h * attn_weights

        # 残差连接
        output = h_attended + residual

        return output

# 最终模型（用于BRIGHT数据集）
class mmscd_siam_GCN_WT_singleSTM(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_WT_singleSTM, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.WT_opt = WTFMBlock(3, 16, wavelet_type='db4')
        self.WT_sar = WTFMBlock(3, 16)
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.GCN1 = GCN(80, 80)
        self.GCN2 = GCN(160, 160)
        self.GCN3 = GCN(320, 320)
        self.GCN4 = GCN(640, 640)
        self.GCN5 = GCN(640, 640)
        self.GCN = [self.GCN1, self.GCN2, self.GCN3, self.GCN4, self.GCN5]
        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)

        else:
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.STMambaLayer = STM_GrootV3D_V2(640, False)
        self.softmax = nn.Softmax(dim=1)

        # 添加自适应融合模块
        self.fusion_0 = AdaptiveFusion(self.channel_nums[0])
        self.fusion_1 = AdaptiveFusion(self.channel_nums[1])
        self.fusion_2 = AdaptiveFusion(self.channel_nums[2])
        self.fusion_3 = AdaptiveFusion(self.channel_nums[3])
        self.fusion_4 = AdaptiveFusion(self.channel_nums[4])
        self.fusions = [self.fusion_0, self.fusion_1, self.fusion_2, self.fusion_3, self.fusion_4]

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]

        self.classifier = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.CD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
#dongjie bufencanshu
        # # 加载预训练权重（关键修改）
        # if pretrained:
        #     pretrained_weights = torch.load(
        #         '/mnt/nas/checkpoints/BRIGHT/Proposed-1.3_baseline_WT_CIEM_GCN_FUSION_20260103_203418/best_model_val_68.5_test_66.6.pth')
        #     components_map = {
        #         'WT_opt.': self.WT_opt,
        #         'WT_sar.': self.WT_sar,
        #         'backbone.': self.backbone}
        #     for prefix, component in components_map.items():
        #         updated_weights = {
        #             key.replace(prefix, ''): value
        #             for key, value in pretrained_weights.items()
        #             if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()}
        #         if updated_weights:
        #             component.load_state_dict(updated_weights, strict=True)
        #             print(
        #                 f'Successfully loaded {prefix.strip(".")} pre-training weights! ({len(updated_weights)} layers)')
        #         else:
        #             print(f'No matching weights found for {prefix.strip(".")}')
        #     for component in components_map.values():
        #         for param in component.parameters():
        #             param.requires_grad = False
        #     print('All specified components have been frozen.')
        # else:
        #     self.load_adapted_pretrained_weights()
        #     for param in self.backbone.parameters():
        #         param.requires_grad = True
        # initialize_weights(self.CD_Decoder, self.CD, self.classifier)

# all no frozen
        # 加载预训练权重（关键修改）
        if pretrained:
            # 加载完整模型权重
            checkpoint = torch.load(
                '/mnt/nas/checkpoints/BRIGHT/best_model_val_69.5_test_67.7.pth')

            # 检查检查点结构
            if isinstance(checkpoint, dict):
                # 如果检查点包含'state_dict'键，则提取它
                if 'state_dict' in checkpoint:
                    pretrained_weights = checkpoint['state_dict']
                    print("✅ 检测到包含state_dict的检查点")
                else:
                    # 否则假设整个字典就是权重
                    pretrained_weights = checkpoint
                    print("✅ 加载完整模型权重")

                # 尝试加载完整模型权重
                try:
                    # 严格模式设为False，允许部分权重不匹配
                    missing_keys, unexpected_keys = self.load_state_dict(pretrained_weights, strict=False)

                    print(f'✅ 成功加载预训练权重！')

                    if missing_keys:
                        print(f'⚠️ 缺失的键 ({len(missing_keys)} 个):')
                        for key in list(missing_keys)[:10]:  # 只显示前10个
                            print(f'  - {key}')
                        if len(missing_keys) > 10:
                            print(f'  ... 还有 {len(missing_keys) - 10} 个')

                    if unexpected_keys:
                        print(f'⚠️ 意外的键 ({len(unexpected_keys)} 个):')
                        for key in list(unexpected_keys)[:10]:  # 只显示前10个
                            print(f'  - {key}')
                        if len(unexpected_keys) > 10:
                            print(f'  ... 还有 {len(unexpected_keys) - 10} 个')

                except Exception as e:
                    print(f'❌ 加载完整模型权重失败: {e}')
                    print('🔄 尝试回退到组件级加载...')

                    # 回退到原来的组件级加载
                    components_map = {
                        'WT_opt.': self.WT_opt,
                        'WT_sar.': self.WT_sar,
                        'backbone.': self.backbone,
                        'GCN1.': self.GCN1,
                        'GCN2.': self.GCN2,
                        'GCN3.': self.GCN3,
                        'GCN4.': self.GCN4,
                        'GCN5.': self.GCN5,
                        'CFEM_0.': self.CFEM_0,
                        'CFEM_1.': self.CFEM_1,
                        'CFEM_2.': self.CFEM_2,
                        'CFEM_3.': self.CFEM_3,
                        'CFEM_4.': self.CFEM_4,
                        'fusion_0.': self.fusion_0,
                        'fusion_1.': self.fusion_1,
                        'fusion_2.': self.fusion_2,
                        'fusion_3.': self.fusion_3,
                        'fusion_4.': self.fusion_4,
                        'CD_Decoder.': self.CD_Decoder,
                        'CD.': self.CD,
                        'classifier.': self.classifier
                    }

                    for prefix, component in components_map.items():
                        updated_weights = {
                            key.replace(prefix, ''): value
                            for key, value in pretrained_weights.items()
                            if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()
                        }
                        if updated_weights:
                            component.load_state_dict(updated_weights, strict=False)
                            print(f'✅ 加载 {prefix.strip(".")} 权重 ({len(updated_weights)} 层)')
                        else:
                            print(f'⚠️ 未找到 {prefix.strip(".")} 的匹配权重')

            else:
                print("❌ 检查点格式未知")

            # 重要：不冻结参数，全部设置为可训练
            for param in self.parameters():
                param.requires_grad = True
            print('✅ 所有参数已设置为可训练状态（继续训练模式）')

            # 如果检查点包含优化器状态和训练状态，可以在这里记录
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    print(f'📅 检查点来自 epoch: {checkpoint["epoch"]}')
                if 'best_val_score' in checkpoint:
                    print(f'🏆 最佳验证分数: {checkpoint["best_val_score"]:.4f}')
                if 'optimizer' in checkpoint:
                    print('⚙️ 检查点包含优化器状态')
                if 'scheduler' in checkpoint:
                    print('📈 检查点包含学习率调度器状态')

        else:
            self.load_adapted_pretrained_weights()
            for param in self.backbone.parameters():
                param.requires_grad = True
            print('✅ 加载ImageNet预训练权重，backbone可训练')

        # 初始化未被加载的权重
        initialize_weights(self.CD_Decoder, self.CD, self.classifier)

    def load_adapted_pretrained_weights(self):
        """智能加载适配的预训练权重"""
        try:
            # 加载预训练权重
            pretrained_weights = torch.load('/home/remote/Liyujie-daima/SGF-Net-master/grootv_cls_tiny.pth')
            new_dict = pretrained_weights['model']

            # 创建适配的权重字典
            updated_weights = {}
            missing_keys = []
            unexpected_keys = []

            # 获取当前模型状态字典
            model_dict = self.backbone.state_dict()

            for key, value in new_dict.items():
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        # 形状完全匹配，直接加载
                        updated_weights[key] = value
                    elif key == 'patch_embed.conv1.weight':
                        # 特殊处理输入卷积层权重适配
                        adapted_weight = self.adapt_input_conv_weights(value, model_dict[key].shape)
                        updated_weights[key] = adapted_weight
                        print(f"✅ 适配输入层权重: {value.shape} -> {adapted_weight.shape}")
                    else:
                        # 其他不匹配的层，跳过并记录
                        missing_keys.append(key)
                        print(f"⚠️ 跳过不匹配的层: {key} {value.shape} -> {model_dict[key].shape}")
                else:
                    unexpected_keys.append(key)

            # 加载适配后的权重
            self.backbone.load_state_dict(updated_weights, strict=False)

            # 打印加载结果
            print("🎯 预训练权重加载完成!")
            print(f"📊 成功加载: {len(updated_weights)}/{len(model_dict)} 层")
            if missing_keys:
                print(f"⚠️ 跳过的层: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"❌ 未使用的预训练层: {len(unexpected_keys)} 个")

        except Exception as e:
            print(f"❌ 预训练权重加载失败: {e}")

    def adapt_input_conv_weights(self, original_weight, target_shape):
        """
        适配输入卷积层权重
        将3通道权重扩展到32通道
        """
        out_channels, in_channels, kh, kw = target_shape
        new_weight = torch.zeros(target_shape)

        # 计算重复因子
        repeat_factor = in_channels // original_weight.size(1)

        if repeat_factor > 0:
            # 将原始3通道权重复制到新的输入通道
            for i in range(repeat_factor):
                start_ch = i * original_weight.size(1)
                end_ch = (i + 1) * original_weight.size(1)
                if end_ch <= in_channels:
                    # 平均分配权重，保持数值稳定性
                    new_weight[:, start_ch:end_ch] = original_weight / repeat_factor

            # 处理剩余的通道（如果有）
            remainder = in_channels % original_weight.size(1)
            if remainder > 0:
                start_ch = repeat_factor * original_weight.size(1)
                new_weight[:, start_ch:start_ch + remainder] = original_weight[:, :remainder] / (repeat_factor + 1)
        else:
            # 如果目标通道数小于原始通道数，取前n个通道
            new_weight = original_weight[:, :in_channels] * (in_channels / original_weight.size(1))

        return new_weight

        # 定义要加载的组件映射
        components_map = {
            'WT_opt.': self.WT_opt,
            'WT_sar.': self.WT_sar,
            'backbone.': self.backbone}

        # 统一处理所有组件
        for prefix, component in components_map.items():
            # 使用字典推导式过滤和重命名权重
            updated_weights = {
                key.replace(prefix, ''): value
                for key, value in pretrained_weights.items()
                if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()
            }

            if updated_weights:
                component.load_state_dict(updated_weights, strict=True)
                print(f'Successfully loaded {prefix.strip(".")} pre-training weights! ({len(updated_weights)} layers)')
            else:
                print(f'No matching weights found for {prefix.strip(".")}')

        # 统一冻结参数
        for component in components_map.values():
            for param in component.parameters():
                param.requires_grad = True
        print('All specified components have been frozen.param.requires_grad is True')

        initialize_weights(self.CD_Decoder, self.classifierCD, self.seg)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, opt, sar):
        opt = self.WT_opt(opt)
        sar = self.WT_sar(sar)
        b, c, h, w = opt.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = opt
        xy_in[:, :, :, w:2*w] = sar
        feature_xy = self.backbone.forward(xy_in)

        # 遍历A中的每个矩阵
        feature1 = []
        feature2 = []

        for matrix in feature_xy:
            # 在W维度上划分
            Bs, Cs, Hs, Ws = matrix.shape
            Ws = Ws//2
            T1_part = matrix[:, :, :, 0:Ws]  # 左半部分
            T2_part = matrix[:, :, :, Ws:2*Ws]  # 右半部分
            # 将各部分分别存储到feature列表中
            feature1.append(T1_part)
            feature2.append(T2_part)

        feature1_region = feature1
        feature2_region = feature2

        xf_sm1, yf_sm1 = self.STMambaLayer(feature1[-1], feature2[-1])
        yf_sm2, xf_sm2 = self.STMambaLayer(feature2[-1], feature1[-1])
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        feature1[-1] = x_f
        feature2[-1] = y_f

        feature1_c = []
        feature2_c = []
        # 遍历所有层级的特征
        for i in range(len(feature1)):
            # 获取当前层级的特征
            f1 = feature1[i]
            f2 = feature2[i]

            # 动态获取当前层级维度信息
            batch_size, channels, height, width = f1.size()

            gcn_f1 = f1.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f1 = self.GCN[i](gcn_f1)
            maps_f1 = map_f1.contiguous().view(batch_size, self.channel_nums[i], *f1.size()[2:])

            gcn_f2 = f2.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f2 = self.GCN[i](gcn_f2)
            maps_f2 = map_f2.contiguous().view(batch_size, self.channel_nums[i], *f2.size()[2:])

            # 自适应特征融合 - 替换原有的简单相加操作
            restored_f1 = self.fusions[i](feature1_region[i], maps_f1)
            restored_f2 = self.fusions[i](feature2_region[i], maps_f2)

            # 更新特征列表
            feature1_c.append(restored_f1)
            feature2_c.append(restored_f2)

        # 共性特异性特征融合
        feature_diff = []
        for i in range(len(feature1_c)):
            feature_diff.append(self.CFEM[i](feature1_c[i], feature2_c[i]))

        xc = self.CD_Decoder(feature_diff)
        bcd = self.CD(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifier(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.softmax(scd)
        bcd = torch.sigmoid(bcd)
        return scd.squeeze(1), bcd.squeeze(1)

# class mmscd_siam_GCN_WT_singleSTM(nn.Module):
#     def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
#         super(mmscd_siam_GCN_WT_singleSTM, self).__init__()
#         self.backbone_name = backbone
#         self.nclass = nclass
#         self.lightweight = lightweight
#         self.M = M
#         self.Lambda = Lambda
#         self.WT_opt = WTFMBlock(3, 16, wavelet_type='db4')
#         self.WT_sar = WTFMBlock(3, 16)
#         self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
#         self.GCN1 = GCN(80, 80)
#         self.GCN2 = GCN(160, 160)
#         self.GCN3 = GCN(320, 320)
#         self.GCN4 = GCN(640, 640)
#         self.GCN5 = GCN(640, 640)
#         self.GCN = [self.GCN1, self.GCN2, self.GCN3, self.GCN4, self.GCN5]
#         if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
#             self.channel_nums = [80, 160, 320, 640, 640]
#         elif backbone == "resnet50":
#             self.channel_nums = [256, 512, 1024, 2048]
#
#         if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
#             self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)
#
#         else:
#             self.CD_Decoder = CD_Decoder(self.channel_nums)
#
#         self.STMambaLayer = STM_GrootV3D_V2(640, False)
#         self.softmax = nn.Softmax(dim=1)
#
#         # 添加自适应融合模块
#         self.fusion_0 = AdaptiveFusion(self.channel_nums[0])
#         self.fusion_1 = AdaptiveFusion(self.channel_nums[1])
#         self.fusion_2 = AdaptiveFusion(self.channel_nums[2])
#         self.fusion_3 = AdaptiveFusion(self.channel_nums[3])
#         self.fusion_4 = AdaptiveFusion(self.channel_nums[4])
#         self.fusions = [self.fusion_0, self.fusion_1, self.fusion_2, self.fusion_3, self.fusion_4]
#
#         self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
#         self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
#         self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
#         self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
#         self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
#         self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]
#
#         self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
#                                           nn.Conv2d(64, 4, kernel_size=1))
#         self.seg = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
#                                           nn.Conv2d(64, 1, kernel_size=1))
#
#         # # 加载预训练权重（关键修改）
#         # pretrained_weights = torch.load('/mnt/nas/Proposed_BRIGHT_best_model_val_69.5_test_67.7.pth')
#         #
#         # # 定义要加载的组件映射
#         # components_map = {
#         #     'WT_opt.': self.WT_opt,
#         #     'WT_sar.': self.WT_sar,
#         #     'backbone.': self.backbone}
#         #
#         # # 统一处理所有组件
#         # for prefix, component in components_map.items():
#         #     # 使用字典推导式过滤和重命名权重
#         #     updated_weights = {
#         #         key.replace(prefix, ''): value
#         #         for key, value in pretrained_weights.items()
#         #         if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()
#         #     }
#         #
#         #     if updated_weights:
#         #         component.load_state_dict(updated_weights, strict=True)
#         #         print(f'Successfully loaded {prefix.strip(".")} pre-training weights! ({len(updated_weights)} layers)')
#         #     else:
#         #         print(f'No matching weights found for {prefix.strip(".")}')
#         #
#         # # 统一冻结参数
#         # for component in components_map.values():
#         #     for param in component.parameters():
#         #         param.requires_grad = False
#         # print('All specified components have been frozen.')
#
#         initialize_weights(self.CD_Decoder, self.classifierCD, self.seg)
#
#
#
#     def _make_layer(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes:
#             downsample = nn.Sequential(
#                 conv1x1(inplanes, planes, stride),
#                 nn.BatchNorm2d(planes))
#
#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, opt, sar):
#         opt = self.WT_opt(opt)
#         sar = self.WT_sar(sar)
#         b, c, h, w = opt.shape
#         # features extraction from HR images
#         xy_in = torch.empty(b, c, h, 2 * w).cuda()
#         xy_in[:, :, :, 0:w] = opt
#         xy_in[:, :, :, w:2*w] = sar
#         feature_xy = self.backbone.forward(xy_in)
#
#         # 遍历A中的每个矩阵
#         feature1 = []
#         feature2 = []
#
#         for matrix in feature_xy:
#             # 在W维度上划分
#             Bs, Cs, Hs, Ws = matrix.shape
#             Ws = Ws//2
#             T1_part = matrix[:, :, :, 0:Ws]  # 左半部分
#             T2_part = matrix[:, :, :, Ws:2*Ws]  # 右半部分
#             # 将各部分分别存储到feature列表中
#             feature1.append(T1_part)
#             feature2.append(T2_part)
#
#         feature1_region = feature1
#         feature2_region = feature2
#
#         xf_sm1, yf_sm1 = self.STMambaLayer(feature1[-1], feature2[-1])
#         yf_sm2, xf_sm2 = self.STMambaLayer(feature2[-1], feature1[-1])
#         x_f = xf_sm1 + xf_sm2
#         y_f = yf_sm1 + yf_sm2
#         feature1[-1] = x_f
#         feature2[-1] = y_f
#
#         feature1_c = []
#         feature2_c = []
#         # 遍历所有层级的特征
#         for i in range(len(feature1)):
#             # 获取当前层级的特征
#             f1 = feature1[i]
#             f2 = feature2[i]
#
#             # 动态获取当前层级维度信息
#             batch_size, channels, height, width = f1.size()
#
#             gcn_f1 = f1.contiguous().view(batch_size, self.channel_nums[i], -1)
#             map_f1 = self.GCN[i](gcn_f1)
#             maps_f1 = map_f1.contiguous().view(batch_size, self.channel_nums[i], *f1.size()[2:])
#
#             gcn_f2 = f2.contiguous().view(batch_size, self.channel_nums[i], -1)
#             map_f2 = self.GCN[i](gcn_f2)
#             maps_f2 = map_f2.contiguous().view(batch_size, self.channel_nums[i], *f2.size()[2:])
#
#             # 自适应特征融合 - 替换原有的简单相加操作
#             restored_f1 = self.fusions[i](feature1_region[i], maps_f1)
#             restored_f2 = self.fusions[i](feature2_region[i], maps_f2)
#
#             # 更新特征列表
#             feature1_c.append(restored_f1)
#             feature2_c.append(restored_f2)
#
#         # 共性特异性特征融合
#         feature_diff = []
#         for i in range(len(feature1_c)):
#             feature_diff.append(self.CFEM[i](feature1_c[i], feature2_c[i]))
#
#         xc = self.CD_Decoder(feature_diff)
#         bcd = self.seg(xc)
#         bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
#         scd = self.classifierCD(xc)
#         scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
#         scd = self.softmax(scd)
#         bcd = torch.sigmoid(bcd)
#         return scd.squeeze(1), bcd.squeeze(1)



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
        对输入特征图进行二维小波变换 - 自动适应不同滤波器长度
        """
        B, C, H, W = x.shape

        # 根据滤波器长度计算需要的填充
        filter_length = self.ll.shape[-1]
        padding_needed = filter_length - 1

        # 确保输入尺寸足够大
        if H < filter_length or W < filter_length:
            # 上采样到最小尺寸
            new_H = max(H, filter_length)
            new_W = max(W, filter_length)
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
            B, C, H, W = x.shape

        # 应用对称填充
        if padding_needed > 0:
            pad_size = padding_needed // 2
            x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        else:
            x_padded = x

        # 扩展滤波器到正确的通道数
        ll = self.ll.expand(C, 1, 1, -1).to(x.device)
        lh = self.lh.expand(C, 1, 1, -1).to(x.device)
        hl = self.hl.expand(C, 1, -1, 1).to(x.device)
        hh = self.hh.expand(C, 1, -1, 1).to(x.device)

        # 在行方向进行卷积（下采样2倍）
        x_low_row = F.conv2d(x_padded, ll, stride=(1, 2), padding=(0, 0), groups=C)
        x_high_row = F.conv2d(x_padded, lh, stride=(1, 2), padding=(0, 0), groups=C)

        # 在列方向进行卷积（下采样2倍）
        cA = F.conv2d(x_low_row, hl, stride=(2, 1), padding=(0, 0), groups=C)
        cH = F.conv2d(x_low_row, hh, stride=(2, 1), padding=(0, 0), groups=C)
        cV = F.conv2d(x_high_row, hl, stride=(2, 1), padding=(0, 0), groups=C)
        cD = F.conv2d(x_high_row, hh, stride=(2, 1), padding=(0, 0), groups=C)

        return cA, cH, cV, cD

# 最终模型（用于SN6新数据集）
class mmscd_siam_GCN_WT_singleSTM_SN6(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_WT_singleSTM_SN6, self).__init__()
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda

        self.WT_opt = WTFMBlock(3, 16, wavelet_type='db4')
        self.WT_sar = WTFMBlock(3, 16)
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.GCN1 = GCN(80, 80)
        self.GCN2 = GCN(160, 160)
        self.GCN3 = GCN(320, 320)
        self.GCN4 = GCN(640, 640)
        self.GCN5 = GCN(640, 640)
        self.GCN = [self.GCN1, self.GCN2, self.GCN3, self.GCN4, self.GCN5]
        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)

        else:
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        # self.STMambaLayer = STM_GrootV3D_V2(640, False)
        self.softmax = nn.Softmax(dim=1)

        # 添加自适应融合模块
        self.fusion_0 = AdaptiveFusion(self.channel_nums[0])
        self.fusion_1 = AdaptiveFusion(self.channel_nums[1])
        self.fusion_2 = AdaptiveFusion(self.channel_nums[2])
        self.fusion_3 = AdaptiveFusion(self.channel_nums[3])
        self.fusion_4 = AdaptiveFusion(self.channel_nums[4])
        self.fusions = [self.fusion_0, self.fusion_1, self.fusion_2, self.fusion_3, self.fusion_4]

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]


        self.classifier = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 7, kernel_size=1))
        self.CD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        # 加载预训练权重（关键修改）
        if pretrained:
            # pretrained_weights = torch.load(
            #     '/root/autodl-fs/SN_6/SN6_mmscd_siam_GCN_WT_singleSTM_SN6_bs8_all-retrain_best_model_val_87.0_test_87.2.pth')
            pretrained_weights = torch.load('/home/remote/下载/best_model_val_88.6_test_88.5.pth')
            components_map = {
                'WT_opt.': self.WT_opt,
                'WT_sar.': self.WT_sar,
                'backbone.': self.backbone}
            for prefix, component in components_map.items():
                updated_weights = {
                    key.replace(prefix, ''): value
                    for key, value in pretrained_weights.items()
                    if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()}
                if updated_weights:
                    component.load_state_dict(updated_weights, strict=True)
                    print(
                        f'Successfully loaded {prefix.strip(".")} pre-training weights! ({len(updated_weights)} layers)')
                else:
                    print(f'No matching weights found for {prefix.strip(".")}')
            for component in components_map.values():
                for param in component.parameters():
                    param.requires_grad = True
            print('All specified components have been frozen.')
        else:
            self.load_adapted_pretrained_weights()
            for param in self.backbone.parameters():
                param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.CD, self.classifier)

    def load_adapted_pretrained_weights(self):
        """智能加载适配的预训练权重"""
        try:
            # 加载预训练权重
            pretrained_weights = torch.load('/home/remote/下载/best_model_val_88.6_test_88.5.pth')
            new_dict = pretrained_weights['model']

            # 创建适配的权重字典
            updated_weights = {}
            missing_keys = []
            unexpected_keys = []

            # 获取当前模型状态字典
            model_dict = self.backbone.state_dict()

            for key, value in new_dict.items():
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        # 形状完全匹配，直接加载
                        updated_weights[key] = value
                    elif key == 'patch_embed.conv1.weight':
                        # 特殊处理输入卷积层权重适配
                        adapted_weight = self.adapt_input_conv_weights(value, model_dict[key].shape)
                        updated_weights[key] = adapted_weight
                        print(f"✅ 适配输入层权重: {value.shape} -> {adapted_weight.shape}")
                    else:
                        # 其他不匹配的层，跳过并记录
                        missing_keys.append(key)
                        print(f"⚠️ 跳过不匹配的层: {key} {value.shape} -> {model_dict[key].shape}")
                else:
                    unexpected_keys.append(key)

            # 加载适配后的权重
            self.backbone.load_state_dict(updated_weights, strict=False)

            # 打印加载结果
            print("🎯 预训练权重加载完成!")
            print(f"📊 成功加载: {len(updated_weights)}/{len(model_dict)} 层")
            if missing_keys:
                print(f"⚠️ 跳过的层: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"❌ 未使用的预训练层: {len(unexpected_keys)} 个")

        except Exception as e:
            print(f"❌ 预训练权重加载失败: {e}")

    def adapt_input_conv_weights(self, original_weight, target_shape):
        """
        适配输入卷积层权重
        将3通道权重扩展到32通道
        """
        out_channels, in_channels, kh, kw = target_shape
        new_weight = torch.zeros(target_shape)

        # 计算重复因子
        repeat_factor = in_channels // original_weight.size(1)

        if repeat_factor > 0:
            # 将原始3通道权重复制到新的输入通道
            for i in range(repeat_factor):
                start_ch = i * original_weight.size(1)
                end_ch = (i + 1) * original_weight.size(1)
                if end_ch <= in_channels:
                    # 平均分配权重，保持数值稳定性
                    new_weight[:, start_ch:end_ch] = original_weight / repeat_factor

            # 处理剩余的通道（如果有）
            remainder = in_channels % original_weight.size(1)
            if remainder > 0:
                start_ch = repeat_factor * original_weight.size(1)
                new_weight[:, start_ch:start_ch + remainder] = original_weight[:, :remainder] / (repeat_factor + 1)
        else:
            # 如果目标通道数小于原始通道数，取前n个通道
            new_weight = original_weight[:, :in_channels] * (in_channels / original_weight.size(1))

        return new_weight

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, opt, sar):
        opt = self.WT_opt(opt)
        sar = self.WT_sar(sar)
        b, c, h, w = opt.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = opt
        xy_in[:, :, :, w:2 * w] = sar
        feature_xy = self.backbone.forward(xy_in)

        # 遍历A中的每个矩阵
        feature1 = []
        feature2 = []

        for matrix in feature_xy:
            # 在W维度上划分
            Bs, Cs, Hs, Ws = matrix.shape
            Ws = Ws // 2
            T1_part = matrix[:, :, :, 0:Ws]  # 左半部分
            T2_part = matrix[:, :, :, Ws:2 * Ws]  # 右半部分
            # 将各部分分别存储到feature列表中
            feature1.append(T1_part)
            feature2.append(T2_part)

        feature1_region = feature1
        feature2_region = feature2

        # xf_sm1, yf_sm1 = self.STMambaLayer(feature1[-1], feature2[-1])
        # yf_sm2, xf_sm2 = self.STMambaLayer(feature2[-1], feature1[-1])
        # x_f = xf_sm1 + xf_sm2
        # y_f = yf_sm1 + yf_sm2
        # feature1[-1] = x_f
        # feature2[-1] = y_f

        feature1_c = []
        feature2_c = []
        # 遍历所有层级的特征
        for i in range(len(feature1)):
            # 获取当前层级的特征
            f1 = feature1[i]
            f2 = feature2[i]

            # 动态获取当前层级维度信息
            batch_size, channels, height, width = f1.size()

            gcn_f1 = f1.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f1 = self.GCN[i](gcn_f1)
            maps_f1 = map_f1.contiguous().view(batch_size, self.channel_nums[i], *f1.size()[2:])

            gcn_f2 = f2.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f2 = self.GCN[i](gcn_f2)
            maps_f2 = map_f2.contiguous().view(batch_size, self.channel_nums[i], *f2.size()[2:])

            # 自适应特征融合 - 替换原有的简单相加操作
            restored_f1 = self.fusions[i](feature1_region[i], maps_f1)
            restored_f2 = self.fusions[i](feature2_region[i], maps_f2)

            # 更新特征列表
            feature1_c.append(restored_f1)
            feature2_c.append(restored_f2)

        # 共性特异性特征融合
        feature_diff = []
        for i in range(len(feature1_c)):
            feature_diff.append(self.CFEM[i](feature1_c[i], feature2_c[i]))

        xc = self.CD_Decoder(feature_diff)
        bcd = self.CD(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifier(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.softmax(scd)
        bcd = torch.sigmoid(bcd)
        return bcd.squeeze(1), scd.squeeze(1)




# 最终模型（用于Wuhan数据集）
class mmscd_siam_GCN_WT_singleSTM_WH(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_WT_singleSTM_WH, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.WT_opt = WTFMBlock(3, 16, wavelet_type='db4')
        self.WT_sar = WTFMBlock(3, 16)
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.GCN1 = GCN(80, 80)
        self.GCN2 = GCN(160, 160)
        self.GCN3 = GCN(320, 320)
        self.GCN4 = GCN(640, 640)
        self.GCN5 = GCN(640, 640)
        self.GCN = [self.GCN1, self.GCN2, self.GCN3, self.GCN4, self.GCN5]
        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)

        else:
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.STMambaLayer = STM_GrootV3D_V2(640, False)
        self.softmax = nn.Softmax(dim=1)

        # 添加自适应融合模块
        self.fusion_0 = AdaptiveFusion(self.channel_nums[0])
        self.fusion_1 = AdaptiveFusion(self.channel_nums[1])
        self.fusion_2 = AdaptiveFusion(self.channel_nums[2])
        self.fusion_3 = AdaptiveFusion(self.channel_nums[3])
        self.fusion_4 = AdaptiveFusion(self.channel_nums[4])
        self.fusions = [self.fusion_0, self.fusion_1, self.fusion_2, self.fusion_3, self.fusion_4]

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]

        self.CD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        # 加载预训练权重（关键修改）
        if pretrained:
            pretrained_weights = torch.load(
                '/mnt/nas/original_weight/wuhan_best_model_val_63.8_test_64.8.pth')
            components_map = {
                'WT_opt.': self.WT_opt,
                'WT_sar.': self.WT_sar,
                'backbone.': self.backbone}
            for prefix, component in components_map.items():
                updated_weights = {
                    key.replace(prefix, ''): value
                    for key, value in pretrained_weights.items()
                    if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()}
                if updated_weights:
                    component.load_state_dict(updated_weights, strict=True)
                    print(f'Successfully loaded {prefix.strip(".")} pre-training weights! ({len(updated_weights)} layers)')
                else:
                    print(f'No matching weights found for {prefix.strip(".")}')
            for component in components_map.values():
                for param in component.parameters():
                    param.requires_grad = False
            print('All specified components have been frozen.')
        else:
            self.load_adapted_pretrained_weights()
            for param in self.backbone.parameters():
                param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.CD)

    def load_adapted_pretrained_weights(self):
        """智能加载适配的预训练权重"""
        try:
            # 加载预训练权重
            pretrained_weights = torch.load('/home/remote/Liyujie-daima/SGF-Net-master/grootv_cls_tiny.pth')
            new_dict = pretrained_weights['model']

            # 创建适配的权重字典
            updated_weights = {}
            missing_keys = []
            unexpected_keys = []

            # 获取当前模型状态字典
            model_dict = self.backbone.state_dict()

            for key, value in new_dict.items():
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        # 形状完全匹配，直接加载
                        updated_weights[key] = value
                    elif key == 'patch_embed.conv1.weight':
                        # 特殊处理输入卷积层权重适配
                        adapted_weight = self.adapt_input_conv_weights(value, model_dict[key].shape)
                        updated_weights[key] = adapted_weight
                        print(f"✅ 适配输入层权重: {value.shape} -> {adapted_weight.shape}")
                    else:
                        # 其他不匹配的层，跳过并记录
                        missing_keys.append(key)
                        print(f"⚠️ 跳过不匹配的层: {key} {value.shape} -> {model_dict[key].shape}")
                else:
                    unexpected_keys.append(key)

            # 加载适配后的权重
            self.backbone.load_state_dict(updated_weights, strict=False)

            # 打印加载结果
            print("🎯 预训练权重加载完成!")
            print(f"📊 成功加载: {len(updated_weights)}/{len(model_dict)} 层")
            if missing_keys:
                print(f"⚠️ 跳过的层: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"❌ 未使用的预训练层: {len(unexpected_keys)} 个")

        except Exception as e:
            print(f"❌ 预训练权重加载失败: {e}")

    def adapt_input_conv_weights(self, original_weight, target_shape):
        """
        适配输入卷积层权重
        将3通道权重扩展到32通道
        """
        out_channels, in_channels, kh, kw = target_shape
        new_weight = torch.zeros(target_shape)

        # 计算重复因子
        repeat_factor = in_channels // original_weight.size(1)

        if repeat_factor > 0:
            # 将原始3通道权重复制到新的输入通道
            for i in range(repeat_factor):
                start_ch = i * original_weight.size(1)
                end_ch = (i + 1) * original_weight.size(1)
                if end_ch <= in_channels:
                    # 平均分配权重，保持数值稳定性
                    new_weight[:, start_ch:end_ch] = original_weight / repeat_factor

            # 处理剩余的通道（如果有）
            remainder = in_channels % original_weight.size(1)
            if remainder > 0:
                start_ch = repeat_factor * original_weight.size(1)
                new_weight[:, start_ch:start_ch + remainder] = original_weight[:, :remainder] / (repeat_factor + 1)
        else:
            # 如果目标通道数小于原始通道数，取前n个通道
            new_weight = original_weight[:, :in_channels] * (in_channels / original_weight.size(1))

        return new_weight
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, opt, sar):
        opt = self.WT_opt(opt)
        sar = self.WT_sar(sar)
        b, c, h, w = opt.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = opt
        xy_in[:, :, :, w:2*w] = sar
        feature_xy = self.backbone.forward(xy_in)
        # 遍历A中的每个矩阵
        feature1 = []
        feature2 = []
        for matrix in feature_xy:
            # 在W维度上划分
            Bs, Cs, Hs, Ws = matrix.shape
            Ws = Ws//2
            T1_part = matrix[:, :, :, 0:Ws]  # 左半部分
            T2_part = matrix[:, :, :, Ws:2*Ws]  # 右半部分
            # 将各部分分别存储到feature列表中
            feature1.append(T1_part)
            feature2.append(T2_part)
        feature1_region = feature1
        feature2_region = feature2
        xf_sm1, yf_sm1 = self.STMambaLayer(feature1[-1], feature2[-1])
        yf_sm2, xf_sm2 = self.STMambaLayer(feature2[-1], feature1[-1])
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        feature1[-1] = x_f
        feature2[-1] = y_f
        feature1_c = []
        feature2_c = []
        # 遍历所有层级的特征
        for i in range(len(feature1)):
            # 获取当前层级的特征
            f1 = feature1[i]
            f2 = feature2[i]
            # 动态获取当前层级维度信息
            batch_size, channels, height, width = f1.size()
            gcn_f1 = f1.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f1 = self.GCN[i](gcn_f1)
            maps_f1 = map_f1.contiguous().view(batch_size, self.channel_nums[i], *f1.size()[2:])
            gcn_f2 = f2.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f2 = self.GCN[i](gcn_f2)
            maps_f2 = map_f2.contiguous().view(batch_size, self.channel_nums[i], *f2.size()[2:])
            # 自适应特征融合
            restored_f1 = self.fusions[i](feature1_region[i], maps_f1)
            restored_f2 = self.fusions[i](feature2_region[i], maps_f2)
            feature1_c.append(restored_f1)
            feature2_c.append(restored_f2)
        feature_diff = []
        for i in range(len(feature1_c)):
            feature_diff.append(self.CFEM[i](feature1_c[i], feature2_c[i]))
        xc = self.CD_Decoder(feature_diff)
        bcd = self.CD(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        bcd = torch.sigmoid(bcd)
        return bcd.squeeze(1)



# sion moudel
class mmscd_siam_GCN_WT_singleSTM_concat(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_WT_singleSTM_concat, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.WT_opt = WTFMBlock(3, 16, wavelet_type='db4')
        self.WT_sar = WTFMBlock(3, 16)
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.GCN1 = GCN(80, 80)
        self.GCN2 = GCN(160, 160)
        self.GCN3 = GCN(320, 320)
        self.GCN4 = GCN(640, 640)
        self.GCN5 = GCN(640, 640)
        self.GCN = [self.GCN1, self.GCN2, self.GCN3, self.GCN4, self.GCN5]
        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)

        else:
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.STMambaLayer = STM_GrootV3D_V2(640, False)
        self.softmax = nn.Softmax(dim=1)

        # 添加自适应融合模块
        self.fusion_0 = AdaptiveFusion(self.channel_nums[0])
        self.fusion_1 = AdaptiveFusion(self.channel_nums[1])
        self.fusion_2 = AdaptiveFusion(self.channel_nums[2])
        self.fusion_3 = AdaptiveFusion(self.channel_nums[3])
        self.fusion_4 = AdaptiveFusion(self.channel_nums[4])
        self.fusions = [self.fusion_0, self.fusion_1, self.fusion_2, self.fusion_3, self.fusion_4]

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]

        self.classifier = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.CD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
#dongjie bufencanshu
        # # 加载预训练权重（关键修改）
        # if pretrained:
        #     pretrained_weights = torch.load(
        #         '/mnt/nas/checkpoints/BRIGHT/Proposed-1.3_baseline_WT_CIEM_GCN_FUSION_20260103_203418/best_model_val_68.5_test_66.6.pth')
        #     components_map = {
        #         'WT_opt.': self.WT_opt,
        #         'WT_sar.': self.WT_sar,
        #         'backbone.': self.backbone}
        #     for prefix, component in components_map.items():
        #         updated_weights = {
        #             key.replace(prefix, ''): value
        #             for key, value in pretrained_weights.items()
        #             if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()}
        #         if updated_weights:
        #             component.load_state_dict(updated_weights, strict=True)
        #             print(
        #                 f'Successfully loaded {prefix.strip(".")} pre-training weights! ({len(updated_weights)} layers)')
        #         else:
        #             print(f'No matching weights found for {prefix.strip(".")}')
        #     for component in components_map.values():
        #         for param in component.parameters():
        #             param.requires_grad = False
        #     print('All specified components have been frozen.')
        # else:
        #     self.load_adapted_pretrained_weights()
        #     for param in self.backbone.parameters():
        #         param.requires_grad = True
        # initialize_weights(self.CD_Decoder, self.CD, self.classifier)

# all no frozen
        # 加载预训练权重（关键修改）
        if pretrained:
            # 加载完整模型权重
            checkpoint = torch.load(
                '/mnt/nas/checkpoints/BRIGHT/Proposed-1.3_baseline_WT_CIEM_GCN_FUSION_20260103_203418/best_model_val_68.5_test_66.6.pth')

            # 检查检查点结构
            if isinstance(checkpoint, dict):
                # 如果检查点包含'state_dict'键，则提取它
                if 'state_dict' in checkpoint:
                    pretrained_weights = checkpoint['state_dict']
                    print("✅ 检测到包含state_dict的检查点")
                else:
                    # 否则假设整个字典就是权重
                    pretrained_weights = checkpoint
                    print("✅ 加载完整模型权重")

                # 尝试加载完整模型权重
                try:
                    # 严格模式设为False，允许部分权重不匹配
                    missing_keys, unexpected_keys = self.load_state_dict(pretrained_weights, strict=False)

                    print(f'✅ 成功加载预训练权重！')

                    if missing_keys:
                        print(f'⚠️ 缺失的键 ({len(missing_keys)} 个):')
                        for key in list(missing_keys)[:10]:  # 只显示前10个
                            print(f'  - {key}')
                        if len(missing_keys) > 10:
                            print(f'  ... 还有 {len(missing_keys) - 10} 个')

                    if unexpected_keys:
                        print(f'⚠️ 意外的键 ({len(unexpected_keys)} 个):')
                        for key in list(unexpected_keys)[:10]:  # 只显示前10个
                            print(f'  - {key}')
                        if len(unexpected_keys) > 10:
                            print(f'  ... 还有 {len(unexpected_keys) - 10} 个')

                except Exception as e:
                    print(f'❌ 加载完整模型权重失败: {e}')
                    print('🔄 尝试回退到组件级加载...')

                    # 回退到原来的组件级加载
                    components_map = {
                        'WT_opt.': self.WT_opt,
                        'WT_sar.': self.WT_sar,
                        'backbone.': self.backbone,
                        'GCN1.': self.GCN1,
                        'GCN2.': self.GCN2,
                        'GCN3.': self.GCN3,
                        'GCN4.': self.GCN4,
                        'GCN5.': self.GCN5,
                        'CFEM_0.': self.CFEM_0,
                        'CFEM_1.': self.CFEM_1,
                        'CFEM_2.': self.CFEM_2,
                        'CFEM_3.': self.CFEM_3,
                        'CFEM_4.': self.CFEM_4,
                        'fusion_0.': self.fusion_0,
                        'fusion_1.': self.fusion_1,
                        'fusion_2.': self.fusion_2,
                        'fusion_3.': self.fusion_3,
                        'fusion_4.': self.fusion_4,
                        'CD_Decoder.': self.CD_Decoder,
                        'CD.': self.CD,
                        'classifier.': self.classifier
                    }

                    for prefix, component in components_map.items():
                        updated_weights = {
                            key.replace(prefix, ''): value
                            for key, value in pretrained_weights.items()
                            if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()
                        }
                        if updated_weights:
                            component.load_state_dict(updated_weights, strict=False)
                            print(f'✅ 加载 {prefix.strip(".")} 权重 ({len(updated_weights)} 层)')
                        else:
                            print(f'⚠️ 未找到 {prefix.strip(".")} 的匹配权重')

            else:
                print("❌ 检查点格式未知")

            # 重要：不冻结参数，全部设置为可训练
            for param in self.parameters():
                param.requires_grad = True
            print('✅ 所有参数已设置为可训练状态（继续训练模式）')

            # 如果检查点包含优化器状态和训练状态，可以在这里记录
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    print(f'📅 检查点来自 epoch: {checkpoint["epoch"]}')
                if 'best_val_score' in checkpoint:
                    print(f'🏆 最佳验证分数: {checkpoint["best_val_score"]:.4f}')
                if 'optimizer' in checkpoint:
                    print('⚙️ 检查点包含优化器状态')
                if 'scheduler' in checkpoint:
                    print('📈 检查点包含学习率调度器状态')

        else:
            self.load_adapted_pretrained_weights()
            for param in self.backbone.parameters():
                param.requires_grad = True
            print('✅ 加载ImageNet预训练权重，backbone可训练')

        # 初始化未被加载的权重
        initialize_weights(self.CD_Decoder, self.CD, self.classifier)



    def load_adapted_pretrained_weights(self):
        """智能加载适配的预训练权重"""
        try:
            # 加载预训练权重
            pretrained_weights = torch.load('/home/remote/Liyujie-daima/SGF-Net-master/grootv_cls_tiny.pth')
            new_dict = pretrained_weights['model']

            # 创建适配的权重字典
            updated_weights = {}
            missing_keys = []
            unexpected_keys = []

            # 获取当前模型状态字典
            model_dict = self.backbone.state_dict()

            for key, value in new_dict.items():
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        # 形状完全匹配，直接加载
                        updated_weights[key] = value
                    elif key == 'patch_embed.conv1.weight':
                        # 特殊处理输入卷积层权重适配
                        adapted_weight = self.adapt_input_conv_weights(value, model_dict[key].shape)
                        updated_weights[key] = adapted_weight
                        print(f"✅ 适配输入层权重: {value.shape} -> {adapted_weight.shape}")
                    else:
                        # 其他不匹配的层，跳过并记录
                        missing_keys.append(key)
                        print(f"⚠️ 跳过不匹配的层: {key} {value.shape} -> {model_dict[key].shape}")
                else:
                    unexpected_keys.append(key)

            # 加载适配后的权重
            self.backbone.load_state_dict(updated_weights, strict=False)

            # 打印加载结果
            print("🎯 预训练权重加载完成!")
            print(f"📊 成功加载: {len(updated_weights)}/{len(model_dict)} 层")
            if missing_keys:
                print(f"⚠️ 跳过的层: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"❌ 未使用的预训练层: {len(unexpected_keys)} 个")

        except Exception as e:
            print(f"❌ 预训练权重加载失败: {e}")

    def adapt_input_conv_weights(self, original_weight, target_shape):
        """
        适配输入卷积层权重
        将3通道权重扩展到32通道
        """
        out_channels, in_channels, kh, kw = target_shape
        new_weight = torch.zeros(target_shape)

        # 计算重复因子
        repeat_factor = in_channels // original_weight.size(1)

        if repeat_factor > 0:
            # 将原始3通道权重复制到新的输入通道
            for i in range(repeat_factor):
                start_ch = i * original_weight.size(1)
                end_ch = (i + 1) * original_weight.size(1)
                if end_ch <= in_channels:
                    # 平均分配权重，保持数值稳定性
                    new_weight[:, start_ch:end_ch] = original_weight / repeat_factor

            # 处理剩余的通道（如果有）
            remainder = in_channels % original_weight.size(1)
            if remainder > 0:
                start_ch = repeat_factor * original_weight.size(1)
                new_weight[:, start_ch:start_ch + remainder] = original_weight[:, :remainder] / (repeat_factor + 1)
        else:
            # 如果目标通道数小于原始通道数，取前n个通道
            new_weight = original_weight[:, :in_channels] * (in_channels / original_weight.size(1))

        return new_weight

        # 定义要加载的组件映射
        components_map = {
            'WT_opt.': self.WT_opt,
            'WT_sar.': self.WT_sar,
            'backbone.': self.backbone}

        # 统一处理所有组件
        for prefix, component in components_map.items():
            # 使用字典推导式过滤和重命名权重
            updated_weights = {
                key.replace(prefix, ''): value
                for key, value in pretrained_weights.items()
                if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()
            }

            if updated_weights:
                component.load_state_dict(updated_weights, strict=True)
                print(f'Successfully loaded {prefix.strip(".")} pre-training weights! ({len(updated_weights)} layers)')
            else:
                print(f'No matching weights found for {prefix.strip(".")}')

        # 统一冻结参数
        for component in components_map.values():
            for param in component.parameters():
                param.requires_grad = True
        print('All specified components have been frozen.param.requires_grad is True')

        initialize_weights(self.CD_Decoder, self.classifierCD, self.seg)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, opt, sar):
        opt = self.WT_opt(opt)
        sar = self.WT_sar(sar)
        b, c, h, w = opt.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = opt
        xy_in[:, :, :, w:2*w] = sar
        feature_xy = self.backbone.forward(xy_in)

        # 遍历A中的每个矩阵
        feature1 = []
        feature2 = []

        for matrix in feature_xy:
            # 在W维度上划分
            Bs, Cs, Hs, Ws = matrix.shape
            Ws = Ws//2
            T1_part = matrix[:, :, :, 0:Ws]  # 左半部分
            T2_part = matrix[:, :, :, Ws:2*Ws]  # 右半部分
            # 将各部分分别存储到feature列表中
            feature1.append(T1_part)
            feature2.append(T2_part)

        feature1_region = feature1
        feature2_region = feature2

        xf_sm1, yf_sm1 = self.STMambaLayer(feature1[-1], feature2[-1])
        yf_sm2, xf_sm2 = self.STMambaLayer(feature2[-1], feature1[-1])
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        feature1[-1] = x_f
        feature2[-1] = y_f

        feature1_c = []
        feature2_c = []
        # 遍历所有层级的特征
        for i in range(len(feature1)):
            # 获取当前层级的特征
            f1 = feature1[i]
            f2 = feature2[i]

            # 动态获取当前层级维度信息
            batch_size, channels, height, width = f1.size()

            gcn_f1 = f1.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f1 = self.GCN[i](gcn_f1)
            maps_f1 = map_f1.contiguous().view(batch_size, self.channel_nums[i], *f1.size()[2:])

            gcn_f2 = f2.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f2 = self.GCN[i](gcn_f2)
            maps_f2 = map_f2.contiguous().view(batch_size, self.channel_nums[i], *f2.size()[2:])

            # 自适应特征融合 - 替换原有的简单相加操作
            restored_f1 = self.fusions[i](feature1_region[i], maps_f1)
            restored_f2 = self.fusions[i](feature2_region[i], maps_f2)

            # 更新特征列表
            feature1_c.append(restored_f1)
            feature2_c.append(restored_f2)

        # 共性特异性特征融合
        feature_diff = []
        for i in range(len(feature1_c)):
            feature_diff.append(self.CFEM[i](feature1_c[i], feature2_c[i]))

        xc = self.CD_Decoder(feature_diff)
        bcd = self.CD(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifier(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.softmax(scd)
        bcd = torch.sigmoid(bcd)
        return scd.squeeze(1), bcd.squeeze(1)

# gcn fourstage moudel
class mmscd_siam_GCN_WT_singleSTM_fourstage(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_WT_singleSTM_fourstage, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.WT_opt = WTFMBlock(3, 16, wavelet_type='db4')
        self.WT_sar = WTFMBlock(3, 16)
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        # self.GCN1 = GCN(80, 80)
        # self.GCN2 = GCN(160, 160)
        # self.GCN3 = GCN(320, 320)
        # self.GCN4 = GCN(640, 640)
        self.GCN5 = GCN(640, 640)
        # self.GCN = [self.GCN1, self.GCN2, self.GCN3, self.GCN4, self.GCN5]
        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)

        else:
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.STMambaLayer = STM_GrootV3D_V2(640, False)
        self.softmax = nn.Softmax(dim=1)

        # 添加自适应融合模块
        # self.fusion_0 = AdaptiveFusion(self.channel_nums[0])
        # self.fusion_1 = AdaptiveFusion(self.channel_nums[1])
        # self.fusion_2 = AdaptiveFusion(self.channel_nums[2])
        # self.fusion_3 = AdaptiveFusion(self.channel_nums[3])
        self.fusion_4 = AdaptiveFusion(self.channel_nums[4])
        # self.fusions = [self.fusion_0, self.fusion_1, self.fusion_2, self.fusion_3, self.fusion_4]

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]

        self.classifier = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.CD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
#dongjie bufencanshu
        # # 加载预训练权重（关键修改）
        # if pretrained:
        #     pretrained_weights = torch.load(
        #         '/mnt/nas/checkpoints/BRIGHT/Proposed-1.3_baseline_WT_CIEM_GCN_FUSION_20260103_203418/best_model_val_68.5_test_66.6.pth')
        #     components_map = {
        #         'WT_opt.': self.WT_opt,
        #         'WT_sar.': self.WT_sar,
        #         'backbone.': self.backbone}
        #     for prefix, component in components_map.items():
        #         updated_weights = {
        #             key.replace(prefix, ''): value
        #             for key, value in pretrained_weights.items()
        #             if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()}
        #         if updated_weights:
        #             component.load_state_dict(updated_weights, strict=True)
        #             print(
        #                 f'Successfully loaded {prefix.strip(".")} pre-training weights! ({len(updated_weights)} layers)')
        #         else:
        #             print(f'No matching weights found for {prefix.strip(".")}')
        #     for component in components_map.values():
        #         for param in component.parameters():
        #             param.requires_grad = False
        #     print('All specified components have been frozen.')
        # else:
        #     self.load_adapted_pretrained_weights()
        #     for param in self.backbone.parameters():
        #         param.requires_grad = True
        # initialize_weights(self.CD_Decoder, self.CD, self.classifier)

# all no frozen
        # 加载预训练权重（关键修改）
        if pretrained:
            # 加载完整模型权重
            checkpoint = torch.load(
                '/mnt/nas/checkpoints/BRIGHT/Proposed-1.3_baseline_WT_CIEM_GCN_FUSION_20260103_203418/best_model_val_68.5_test_66.6.pth')

            # 检查检查点结构
            if isinstance(checkpoint, dict):
                # 如果检查点包含'state_dict'键，则提取它
                if 'state_dict' in checkpoint:
                    pretrained_weights = checkpoint['state_dict']
                    print("✅ 检测到包含state_dict的检查点")
                else:
                    # 否则假设整个字典就是权重
                    pretrained_weights = checkpoint
                    print("✅ 加载完整模型权重")

                # 尝试加载完整模型权重
                try:
                    # 严格模式设为False，允许部分权重不匹配
                    missing_keys, unexpected_keys = self.load_state_dict(pretrained_weights, strict=False)

                    print(f'✅ 成功加载预训练权重！')

                    if missing_keys:
                        print(f'⚠️ 缺失的键 ({len(missing_keys)} 个):')
                        for key in list(missing_keys)[:10]:  # 只显示前10个
                            print(f'  - {key}')
                        if len(missing_keys) > 10:
                            print(f'  ... 还有 {len(missing_keys) - 10} 个')

                    if unexpected_keys:
                        print(f'⚠️ 意外的键 ({len(unexpected_keys)} 个):')
                        for key in list(unexpected_keys)[:10]:  # 只显示前10个
                            print(f'  - {key}')
                        if len(unexpected_keys) > 10:
                            print(f'  ... 还有 {len(unexpected_keys) - 10} 个')

                except Exception as e:
                    print(f'❌ 加载完整模型权重失败: {e}')
                    print('🔄 尝试回退到组件级加载...')

                    # 回退到原来的组件级加载
                    components_map = {
                        'WT_opt.': self.WT_opt,
                        'WT_sar.': self.WT_sar,
                        'backbone.': self.backbone,
                        'GCN1.': self.GCN1,
                        'GCN2.': self.GCN2,
                        'GCN3.': self.GCN3,
                        'GCN4.': self.GCN4,
                        'GCN5.': self.GCN5,
                        'CFEM_0.': self.CFEM_0,
                        'CFEM_1.': self.CFEM_1,
                        'CFEM_2.': self.CFEM_2,
                        'CFEM_3.': self.CFEM_3,
                        'CFEM_4.': self.CFEM_4,
                        'fusion_0.': self.fusion_0,
                        'fusion_1.': self.fusion_1,
                        'fusion_2.': self.fusion_2,
                        'fusion_3.': self.fusion_3,
                        'fusion_4.': self.fusion_4,
                        'CD_Decoder.': self.CD_Decoder,
                        'CD.': self.CD,
                        'classifier.': self.classifier
                    }

                    for prefix, component in components_map.items():
                        updated_weights = {
                            key.replace(prefix, ''): value
                            for key, value in pretrained_weights.items()
                            if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()
                        }
                        if updated_weights:
                            component.load_state_dict(updated_weights, strict=False)
                            print(f'✅ 加载 {prefix.strip(".")} 权重 ({len(updated_weights)} 层)')
                        else:
                            print(f'⚠️ 未找到 {prefix.strip(".")} 的匹配权重')

            else:
                print("❌ 检查点格式未知")

            # 重要：不冻结参数，全部设置为可训练
            for param in self.parameters():
                param.requires_grad = True
            print('✅ 所有参数已设置为可训练状态（继续训练模式）')

            # 如果检查点包含优化器状态和训练状态，可以在这里记录
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    print(f'📅 检查点来自 epoch: {checkpoint["epoch"]}')
                if 'best_val_score' in checkpoint:
                    print(f'🏆 最佳验证分数: {checkpoint["best_val_score"]:.4f}')
                if 'optimizer' in checkpoint:
                    print('⚙️ 检查点包含优化器状态')
                if 'scheduler' in checkpoint:
                    print('📈 检查点包含学习率调度器状态')

        else:
            self.load_adapted_pretrained_weights()
            for param in self.backbone.parameters():
                param.requires_grad = True
            print('✅ 加载ImageNet预训练权重，backbone可训练')

        # 初始化未被加载的权重
        initialize_weights(self.CD_Decoder, self.CD, self.classifier)



    def load_adapted_pretrained_weights(self):
        """智能加载适配的预训练权重"""
        try:
            # 加载预训练权重
            pretrained_weights = torch.load('/home/remote/Liyujie-daima/SGF-Net-master/grootv_cls_tiny.pth')
            new_dict = pretrained_weights['model']

            # 创建适配的权重字典
            updated_weights = {}
            missing_keys = []
            unexpected_keys = []

            # 获取当前模型状态字典
            model_dict = self.backbone.state_dict()

            for key, value in new_dict.items():
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        # 形状完全匹配，直接加载
                        updated_weights[key] = value
                    elif key == 'patch_embed.conv1.weight':
                        # 特殊处理输入卷积层权重适配
                        adapted_weight = self.adapt_input_conv_weights(value, model_dict[key].shape)
                        updated_weights[key] = adapted_weight
                        print(f"✅ 适配输入层权重: {value.shape} -> {adapted_weight.shape}")
                    else:
                        # 其他不匹配的层，跳过并记录
                        missing_keys.append(key)
                        print(f"⚠️ 跳过不匹配的层: {key} {value.shape} -> {model_dict[key].shape}")
                else:
                    unexpected_keys.append(key)

            # 加载适配后的权重
            self.backbone.load_state_dict(updated_weights, strict=False)

            # 打印加载结果
            print("🎯 预训练权重加载完成!")
            print(f"📊 成功加载: {len(updated_weights)}/{len(model_dict)} 层")
            if missing_keys:
                print(f"⚠️ 跳过的层: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"❌ 未使用的预训练层: {len(unexpected_keys)} 个")

        except Exception as e:
            print(f"❌ 预训练权重加载失败: {e}")

    def adapt_input_conv_weights(self, original_weight, target_shape):
        """
        适配输入卷积层权重
        将3通道权重扩展到32通道
        """
        out_channels, in_channels, kh, kw = target_shape
        new_weight = torch.zeros(target_shape)

        # 计算重复因子
        repeat_factor = in_channels // original_weight.size(1)

        if repeat_factor > 0:
            # 将原始3通道权重复制到新的输入通道
            for i in range(repeat_factor):
                start_ch = i * original_weight.size(1)
                end_ch = (i + 1) * original_weight.size(1)
                if end_ch <= in_channels:
                    # 平均分配权重，保持数值稳定性
                    new_weight[:, start_ch:end_ch] = original_weight / repeat_factor

            # 处理剩余的通道（如果有）
            remainder = in_channels % original_weight.size(1)
            if remainder > 0:
                start_ch = repeat_factor * original_weight.size(1)
                new_weight[:, start_ch:start_ch + remainder] = original_weight[:, :remainder] / (repeat_factor + 1)
        else:
            # 如果目标通道数小于原始通道数，取前n个通道
            new_weight = original_weight[:, :in_channels] * (in_channels / original_weight.size(1))

        return new_weight

        # 定义要加载的组件映射
        components_map = {
            'WT_opt.': self.WT_opt,
            'WT_sar.': self.WT_sar,
            'backbone.': self.backbone}

        # 统一处理所有组件
        for prefix, component in components_map.items():
            # 使用字典推导式过滤和重命名权重
            updated_weights = {
                key.replace(prefix, ''): value
                for key, value in pretrained_weights.items()
                if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()
            }

            if updated_weights:
                component.load_state_dict(updated_weights, strict=True)
                print(f'Successfully loaded {prefix.strip(".")} pre-training weights! ({len(updated_weights)} layers)')
            else:
                print(f'No matching weights found for {prefix.strip(".")}')

        # 统一冻结参数
        for component in components_map.values():
            for param in component.parameters():
                param.requires_grad = True
        print('All specified components have been frozen.param.requires_grad is True')

        initialize_weights(self.CD_Decoder, self.classifierCD, self.seg)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, opt, sar):
        opt = self.WT_opt(opt)
        sar = self.WT_sar(sar)
        b, c, h, w = opt.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = opt
        xy_in[:, :, :, w:2*w] = sar
        feature_xy = self.backbone.forward(xy_in)

        # 遍历A中的每个矩阵
        feature1 = []
        feature2 = []

        for matrix in feature_xy:
            # 在W维度上划分
            Bs, Cs, Hs, Ws = matrix.shape
            Ws = Ws//2
            T1_part = matrix[:, :, :, 0:Ws]  # 左半部分
            T2_part = matrix[:, :, :, Ws:2*Ws]  # 右半部分
            # 将各部分分别存储到feature列表中
            feature1.append(T1_part)
            feature2.append(T2_part)

        feature1_region = feature1
        feature2_region = feature2

        xf_sm1, yf_sm1 = self.STMambaLayer(feature1[-1], feature2[-1])
        yf_sm2, xf_sm2 = self.STMambaLayer(feature2[-1], feature1[-1])
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        feature1[-1] = x_f
        feature2[-1] = y_f


        feature1_c = feature1
        feature2_c = feature2

        # 只在最后一层应用NonLocal模块
        f1 = feature1[-1]
        f2 = feature2[-1]

        # 动态获取当前层级维度信息
        batch_size, channels, height, width = f1.size()



        gcn_f1 = f1.contiguous().view(batch_size, channels, -1)
        map_f1 = self.GCN5(gcn_f1)
        maps_f1 = map_f1.contiguous().view(batch_size, channels, *f1.size()[2:])

        gcn_f2 = f2.contiguous().view(batch_size, channels, -1)
        map_f2 = self.GCN5(gcn_f2)
        maps_f2 = map_f2.contiguous().view(batch_size, channels, *f2.size()[2:])

        # 自适应特征融合 - 替换原有的简单相加操作
        feature1_c[-1] = self.fusion_4(feature1_region[-1], maps_f1)
        feature2_c[-1] = self.fusion_4(feature2_region[-1], maps_f2)


        # 共性特异性特征融合
        feature_diff = []
        for i in range(len(feature1_c)):
            feature_diff.append(self.CFEM[i](feature1_c[i], feature2_c[i]))

        xc = self.CD_Decoder(feature_diff)
        bcd = self.CD(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifier(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.softmax(scd)
        bcd = torch.sigmoid(bcd)
        return scd.squeeze(1), bcd.squeeze(1)



class mmscd_siam_GCN_WT_singleSTM_concat_11(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_WT_singleSTM_concat_11, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.WT_opt = WTFMBlock(3, 16, wavelet_type='db4')
        self.WT_sar = WTFMBlock(3, 16)
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.GCN1 = GCN(80, 80)
        self.GCN2 = GCN(160, 160)
        self.GCN3 = GCN(320, 320)
        self.GCN4 = GCN(640, 640)
        self.GCN5 = GCN(640, 640)
        self.GCN = [self.GCN1, self.GCN2, self.GCN3, self.GCN4, self.GCN5]
        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_nums = [80, 160, 320, 640, 640]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50" or backbone == "GrootV":
            self.CD_Decoder = Seg_Decoder_ResNet(self.channel_nums)

        else:
            self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.STMambaLayer = STM_GrootV3D_V2(640, False)
        self.softmax = nn.Softmax(dim=1)

        # 添加自适应融合模块
        self.fusion_0 = AdaptiveFusion(self.channel_nums[0])
        self.fusion_1 = AdaptiveFusion(self.channel_nums[1])
        self.fusion_2 = AdaptiveFusion(self.channel_nums[2])
        self.fusion_3 = AdaptiveFusion(self.channel_nums[3])
        self.fusion_4 = AdaptiveFusion(self.channel_nums[4])
        self.fusions = [self.fusion_0, self.fusion_1, self.fusion_2, self.fusion_3, self.fusion_4]

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]

        self.classifier = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.CD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
#dongjie bufencanshu
        # # 加载预训练权重（关键修改）
        if pretrained:
            pretrained_weights = torch.load(
                '/root/autodl-fs/SGF-Net-master/checkpoints/BRIGHT/proposed_method_fourstage1.21_20260121_204658/best_model_val_68.4_test_67.0.pth')
            components_map = {
                'WT_opt.': self.WT_opt,
                'WT_sar.': self.WT_sar,
                'backbone.': self.backbone}
            for prefix, component in components_map.items():
                updated_weights = {
                    key.replace(prefix, ''): value
                    for key, value in pretrained_weights.items()
                    if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()}
                if updated_weights:
                    component.load_state_dict(updated_weights, strict=True)
                    print(
                        f'Successfully loaded {prefix.strip(".")} pre-training weights! ({len(updated_weights)} layers)')
                else:
                    print(f'No matching weights found for {prefix.strip(".")}')
            for component in components_map.values():
                for param in component.parameters():
                    param.requires_grad = False
            print('All specified components have been frozen.')
        else:
            self.load_adapted_pretrained_weights()
            for param in self.backbone.parameters():
                param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.CD, self.classifier)

# # all no frozen
#         # 加载预训练权重（关键修改）
#         if pretrained:
#             # 加载完整模型权重
#             checkpoint = torch.load(
#                 '/mnt/nas/checkpoints/BRIGHT/Proposed-1.3_baseline_WT_CIEM_GCN_FUSION_20260103_203418/best_model_val_68.5_test_66.6.pth')

#             # 检查检查点结构
#             if isinstance(checkpoint, dict):
#                 # 如果检查点包含'state_dict'键，则提取它
#                 if 'state_dict' in checkpoint:
#                     pretrained_weights = checkpoint['state_dict']
#                     print("✅ 检测到包含state_dict的检查点")
#                 else:
#                     # 否则假设整个字典就是权重
#                     pretrained_weights = checkpoint
#                     print("✅ 加载完整模型权重")

#                 # 尝试加载完整模型权重
#                 try:
#                     # 严格模式设为False，允许部分权重不匹配
#                     missing_keys, unexpected_keys = self.load_state_dict(pretrained_weights, strict=False)

#                     print(f'✅ 成功加载预训练权重！')

#                     if missing_keys:
#                         print(f'⚠️ 缺失的键 ({len(missing_keys)} 个):')
#                         for key in list(missing_keys)[:10]:  # 只显示前10个
#                             print(f'  - {key}')
#                         if len(missing_keys) > 10:
#                             print(f'  ... 还有 {len(missing_keys) - 10} 个')

#                     if unexpected_keys:
#                         print(f'⚠️ 意外的键 ({len(unexpected_keys)} 个):')
#                         for key in list(unexpected_keys)[:10]:  # 只显示前10个
#                             print(f'  - {key}')
#                         if len(unexpected_keys) > 10:
#                             print(f'  ... 还有 {len(unexpected_keys) - 10} 个')

#                 except Exception as e:
#                     print(f'❌ 加载完整模型权重失败: {e}')
#                     print('🔄 尝试回退到组件级加载...')

#                     # 回退到原来的组件级加载
#                     components_map = {
#                         'WT_opt.': self.WT_opt,
#                         'WT_sar.': self.WT_sar,
#                         'backbone.': self.backbone,
#                         'GCN1.': self.GCN1,
#                         'GCN2.': self.GCN2,
#                         'GCN3.': self.GCN3,
#                         'GCN4.': self.GCN4,
#                         'GCN5.': self.GCN5,
#                         'CFEM_0.': self.CFEM_0,
#                         'CFEM_1.': self.CFEM_1,
#                         'CFEM_2.': self.CFEM_2,
#                         'CFEM_3.': self.CFEM_3,
#                         'CFEM_4.': self.CFEM_4,
#                         'fusion_0.': self.fusion_0,
#                         'fusion_1.': self.fusion_1,
#                         'fusion_2.': self.fusion_2,
#                         'fusion_3.': self.fusion_3,
#                         'fusion_4.': self.fusion_4,
#                         'CD_Decoder.': self.CD_Decoder,
#                         'CD.': self.CD,
#                         'classifier.': self.classifier
#                     }

#                     for prefix, component in components_map.items():
#                         updated_weights = {
#                             key.replace(prefix, ''): value
#                             for key, value in pretrained_weights.items()
#                             if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()
#                         }
#                         if updated_weights:
#                             component.load_state_dict(updated_weights, strict=False)
#                             print(f'✅ 加载 {prefix.strip(".")} 权重 ({len(updated_weights)} 层)')
#                         else:
#                             print(f'⚠️ 未找到 {prefix.strip(".")} 的匹配权重')

#             else:
#                 print("❌ 检查点格式未知")

#             # 重要：不冻结参数，全部设置为可训练
#             for param in self.parameters():
#                 param.requires_grad = True
#             print('✅ 所有参数已设置为可训练状态（继续训练模式）')

#             # 如果检查点包含优化器状态和训练状态，可以在这里记录
#             if isinstance(checkpoint, dict):
#                 if 'epoch' in checkpoint:
#                     print(f'📅 检查点来自 epoch: {checkpoint["epoch"]}')
#                 if 'best_val_score' in checkpoint:
#                     print(f'🏆 最佳验证分数: {checkpoint["best_val_score"]:.4f}')
#                 if 'optimizer' in checkpoint:
#                     print('⚙️ 检查点包含优化器状态')
#                 if 'scheduler' in checkpoint:
#                     print('📈 检查点包含学习率调度器状态')

#         else:
#             self.load_adapted_pretrained_weights()
#             for param in self.backbone.parameters():
#                 param.requires_grad = True
#             print('✅ 加载ImageNet预训练权重，backbone可训练')

#         # 初始化未被加载的权重
#         initialize_weights(self.CD_Decoder, self.CD, self.classifier)



    def load_adapted_pretrained_weights(self):
        """智能加载适配的预训练权重"""
        try:
            # 加载预训练权重
            pretrained_weights = torch.load('/home/remote/Liyujie-daima/SGF-Net-master/grootv_cls_tiny.pth')
            new_dict = pretrained_weights['model']

            # 创建适配的权重字典
            updated_weights = {}
            missing_keys = []
            unexpected_keys = []

            # 获取当前模型状态字典
            model_dict = self.backbone.state_dict()

            for key, value in new_dict.items():
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        # 形状完全匹配，直接加载
                        updated_weights[key] = value
                    elif key == 'patch_embed.conv1.weight':
                        # 特殊处理输入卷积层权重适配
                        adapted_weight = self.adapt_input_conv_weights(value, model_dict[key].shape)
                        updated_weights[key] = adapted_weight
                        print(f"✅ 适配输入层权重: {value.shape} -> {adapted_weight.shape}")
                    else:
                        # 其他不匹配的层，跳过并记录
                        missing_keys.append(key)
                        print(f"⚠️ 跳过不匹配的层: {key} {value.shape} -> {model_dict[key].shape}")
                else:
                    unexpected_keys.append(key)

            # 加载适配后的权重
            self.backbone.load_state_dict(updated_weights, strict=False)

            # 打印加载结果
            print("🎯 预训练权重加载完成!")
            print(f"📊 成功加载: {len(updated_weights)}/{len(model_dict)} 层")
            if missing_keys:
                print(f"⚠️ 跳过的层: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"❌ 未使用的预训练层: {len(unexpected_keys)} 个")

        except Exception as e:
            print(f"❌ 预训练权重加载失败: {e}")

    def adapt_input_conv_weights(self, original_weight, target_shape):
        """
        适配输入卷积层权重
        将3通道权重扩展到32通道
        """
        out_channels, in_channels, kh, kw = target_shape
        new_weight = torch.zeros(target_shape)

        # 计算重复因子
        repeat_factor = in_channels // original_weight.size(1)

        if repeat_factor > 0:
            # 将原始3通道权重复制到新的输入通道
            for i in range(repeat_factor):
                start_ch = i * original_weight.size(1)
                end_ch = (i + 1) * original_weight.size(1)
                if end_ch <= in_channels:
                    # 平均分配权重，保持数值稳定性
                    new_weight[:, start_ch:end_ch] = original_weight / repeat_factor

            # 处理剩余的通道（如果有）
            remainder = in_channels % original_weight.size(1)
            if remainder > 0:
                start_ch = repeat_factor * original_weight.size(1)
                new_weight[:, start_ch:start_ch + remainder] = original_weight[:, :remainder] / (repeat_factor + 1)
        else:
            # 如果目标通道数小于原始通道数，取前n个通道
            new_weight = original_weight[:, :in_channels] * (in_channels / original_weight.size(1))

        return new_weight

        # 定义要加载的组件映射
        components_map = {
            'WT_opt.': self.WT_opt,
            'WT_sar.': self.WT_sar,
            'backbone.': self.backbone}

        # 统一处理所有组件
        for prefix, component in components_map.items():
            # 使用字典推导式过滤和重命名权重
            updated_weights = {
                key.replace(prefix, ''): value
                for key, value in pretrained_weights.items()
                if key.startswith(prefix) and key.replace(prefix, '') in component.state_dict()
            }

            if updated_weights:
                component.load_state_dict(updated_weights, strict=True)
                print(f'Successfully loaded {prefix.strip(".")} pre-training weights! ({len(updated_weights)} layers)')
            else:
                print(f'No matching weights found for {prefix.strip(".")}')

        # 统一冻结参数
        for component in components_map.values():
            for param in component.parameters():
                param.requires_grad = True
        print('All specified components have been frozen.param.requires_grad is True')

        initialize_weights(self.CD_Decoder, self.classifierCD, self.seg)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, opt, sar):
        opt = self.WT_opt(opt)
        sar = self.WT_sar(sar)
        b, c, h, w = opt.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = opt
        xy_in[:, :, :, w:2*w] = sar
        feature_xy = self.backbone.forward(xy_in)

        # 遍历A中的每个矩阵
        feature1 = []
        feature2 = []

        for matrix in feature_xy:
            # 在W维度上划分
            Bs, Cs, Hs, Ws = matrix.shape
            Ws = Ws//2
            T1_part = matrix[:, :, :, 0:Ws]  # 左半部分
            T2_part = matrix[:, :, :, Ws:2*Ws]  # 右半部分
            # 将各部分分别存储到feature列表中
            feature1.append(T1_part)
            feature2.append(T2_part)

        feature1_region = feature1
        feature2_region = feature2

        xf_sm1, yf_sm1 = self.STMambaLayer(feature1[-1], feature2[-1])
        yf_sm2, xf_sm2 = self.STMambaLayer(feature2[-1], feature1[-1])
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        feature1[-1] = x_f
        feature2[-1] = y_f

        feature1_c = []
        feature2_c = []
        # 遍历所有层级的特征
        for i in range(len(feature1)):
            # 获取当前层级的特征
            f1 = feature1[i]
            f2 = feature2[i]

            # 动态获取当前层级维度信息
            batch_size, channels, height, width = f1.size()

            gcn_f1 = f1.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f1 = self.GCN[i](gcn_f1)
            maps_f1 = map_f1.contiguous().view(batch_size, self.channel_nums[i], *f1.size()[2:])

            gcn_f2 = f2.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f2 = self.GCN[i](gcn_f2)
            maps_f2 = map_f2.contiguous().view(batch_size, self.channel_nums[i], *f2.size()[2:])

            # 自适应特征融合 - 替换原有的简单相加操作
            restored_f1 = self.fusions[i](feature1_region[i], maps_f1)
            restored_f2 = self.fusions[i](feature2_region[i], maps_f2)

            # 更新特征列表
            feature1_c.append(restored_f1)
            feature2_c.append(restored_f2)

        # 共性特异性特征融合
        feature_diff = []
        for i in range(len(feature1_c)):
            feature_diff.append(self.CFEM[i](feature1_c[i], feature2_c[i]))

        xc = self.CD_Decoder(feature_diff)
        bcd = self.CD(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifier(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.softmax(scd)
        bcd = torch.sigmoid(bcd)
        return scd.squeeze(1), bcd.squeeze(1)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MTGrootV3D_SV3(backbone='resnet34', pretrained=True, nclass=7, lightweight=True, M=6, Lambda=0.00005).to(device)
    # model = ST_VSSM_Siam().to(device)
    model = mmscd_siam_GCN_WT_singleSTM(backbone='GrootV', pretrained=False, nclass=4, lightweight=True, M=6, Lambda=0.00005).to(device)
    # print(model)
    image1 = torch.randn(1, 3, 512, 512).to(device)
    MS1 = torch.randn(1, 1, 512, 512).to(device)
    MS1 = MS1.repeat(1, 3, 1, 1)
    bcd = model(image1, MS1)
    # seg1, seg2, seg3, change = model(image1, image2, image3)
    # fs = model(image1, image2, image3, image4, image5, image6, MS1, MS2, MS3, MS4, MS5, MS6)
    # print(seg1)
    from thop import profile
    FLOPs, Params = profile(model, (image1, MS1))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))
