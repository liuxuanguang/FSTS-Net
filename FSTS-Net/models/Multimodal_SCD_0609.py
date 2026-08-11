from models.Backbones.resnet0609 import resnet18, resnet34, resnet50
from models.Decoders.Decoder0609 import Seg_Decoder, CD_Decoder, Seg_Decoder_ResNet, CD_Decoder_ResNet
from models.Modules.CIEM0609 import CIEM
from utils.misc0609 import initialize_weights
from GrootV.classification.models.grootv0609 import GrootVLayer, GrootV3DLayer, MTGrootV3DLayer
from GrootV.classification.models.grootv0609 import GrootV, GrootV_3D
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.WTFMBlock0609 import WTFMBlock

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
        self.GrootV_S1 = GrootV3DLayer(channels=640)
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

class STM_GrootV3D_V2_small(nn.Module):
    def __init__(self, inchannel, channel_first):
        super(STM_GrootV3D_V2_small, self).__init__()
        self.inchannel = inchannel
        self.channel_first = channel_first
        # self.conv2 = nn.Conv2d(kernel_size=1, in_channels=self.inchannel, out_channels=128)
        self.GrootV_S1 = GrootV3DLayer(channels=768)
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

class mmscd_single(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_single, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
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

        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/时空谱联合SCD/TSS-SCD/checkpoints/BRIGHT/mmscd_single_fulldata_20250520_155609/best_model.pth')
        # 1. 过滤出预训练权重中在模型字典中也存在的键
        for key, value in pretrained_weights.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                # 检查新的键是否存在于模型的 state_dict 中
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        # new_dict = pretrained_weights['model']
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        # 防止权重不匹配
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         if key == 'patch_embed.conv1.weight':
        #             # 检查当前模型的 conv1.weight 形状
        #             current_conv1_weight = self.backbone.state_dict()[key]
        #             # 创建一个新的权重，形状与当前模型一致
        #             new_conv1_weight = torch.zeros_like(current_conv1_weight)
        #             # 将预训练权重的前3通道复制到新权重的前3通道
        #             new_conv1_weight[:, :3, :, :] = value
        #             # 将新权重添加到 updated_weights
        #             updated_weights[key] = new_conv1_weight
        #         else:
        #             if key in self.backbone.state_dict():
        #                 updated_weights[key] = value
        #     else:
        #         if key in self.backbone.state_dict():
        #             updated_weights[key] = value

        self.backbone.load_state_dict(updated_weights, strict=False)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded HR Backbone pre-training weights!')

        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD)

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

    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = rgb
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
        
class mmscd_single_STM(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_single_STM, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
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
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/时空谱联合SCD/TSS-SCD/checkpoints/BRIGHT/mmscd_single_fulldata_20250520_155609/best_model.pth')
        # 1. 过滤出预训练权重中在模型字典中也存在的键
        for key, value in pretrained_weights.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                # 检查新的键是否存在于模型的 state_dict 中
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        # new_dict = pretrained_weights['model']
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        # 防止权重不匹配
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         if key == 'patch_embed.conv1.weight':
        #             # 检查当前模型的 conv1.weight 形状
        #             current_conv1_weight = self.backbone.state_dict()[key]
        #             # 创建一个新的权重，形状与当前模型一致
        #             new_conv1_weight = torch.zeros_like(current_conv1_weight)
        #             # 将预训练权重的前3通道复制到新权重的前3通道
        #             new_conv1_weight[:, :3, :, :] = value
        #             # 将新权重添加到 updated_weights
        #             updated_weights[key] = new_conv1_weight
        #         else:
        #             if key in self.backbone.state_dict():
        #                 updated_weights[key] = value
        #     else:
        #         if key in self.backbone.state_dict():
        #             updated_weights[key] = value

        self.backbone.load_state_dict(updated_weights, strict=False)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded HR Backbone pre-training weights!')

        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD)

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


    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = rgb
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

class mmscd_siam(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.rgb_backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.sar_backbone = GrootV_3D(depths=[2, 2, 9, 2])
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
        # self.MambaLayer = STM_MTGrootV3D_V2_DynamicEarth(640, False)
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/New_SECOND_dataset-BiGrootV3D_V1-STM_BiGrootV3D_V2-0114/resnet34/epoch66_Score38.41_mIOU72.90_Sek23.63_Fscd64.16_OA87.67.pth')
        # # 1. 过滤出预训练权重中在模型字典中也存在的键
        # for key, value in pretrained_weights.items():
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         # 检查新的键是否存在于模型的 state_dict 中
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        new_dict = pretrained_weights['model']
        print(pretrained_weights)
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                new_key = key
                if new_key in self.rgb_backbone.state_dict():
                    updated_weights[new_key] = value
        # 防止权重不匹配
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         if key == 'patch_embed.conv1.weight':
        #             # 检查当前模型的 conv1.weight 形状
        #             current_conv1_weight = self.backbone.state_dict()[key]
        #             # 创建一个新的权重，形状与当前模型一致
        #             new_conv1_weight = torch.zeros_like(current_conv1_weight)
        #             # 将预训练权重的前3通道复制到新权重的前3通道
        #             new_conv1_weight[:, :3, :, :] = value
        #             # 将新权重添加到 updated_weights
        #             updated_weights[key] = new_conv1_weight
        #         else:
        #             if key in self.backbone.state_dict():
        #                 updated_weights[key] = value
        #     else:
        #         if key in self.backbone.state_dict():
        #             updated_weights[key] = value

        self.rgb_backbone.load_state_dict(updated_weights, strict=True)
        self.sar_backbone.load_state_dict(updated_weights, strict=True)
        after_weight = self.rgb_backbone.state_dict()
        print('Successfully loaded HR Backbone pre-training weights!')

        for param in self.rgb_backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD)

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

    # def bi_mamba_forward(self, x, y, z, d, s, b):
    #     xf_sm1, yf_sm1, zf_sm1, df_sm1, sf_sm1, bf_sm1 = self.MambaLayer(x, y, z, d, s, b)
    #     bf_sm2, sf_sm2, df_sm2, zf_sm2, yf_sm2, xf_sm2 = self.MambaLayer(b, s, d, z, y, x)
    #     x_f = xf_sm1 + xf_sm2
    #     y_f = yf_sm1 + yf_sm2
    #     z_f = zf_sm1 + zf_sm2
    #     d_f = df_sm1 + df_sm2
    #     s_f = sf_sm1 + sf_sm2
    #     b_f = bf_sm1 + bf_sm2
    #     return x_f, y_f, z_f, d_f, s_f, b_f


    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        feature_rgb = self.rgb_backbone.forward(rgb)
        feature_sar = self.sar_backbone.forward(sar)

        # 遍历A中的每个矩阵
        # feature1_4, feature2_4, feature3_4, feature4_4, feature5_4, feature6_4 = self.bi_mamba_forward(feature1[-1], feature2[-1], feature3[-1],feature4[-1], feature5[-1], feature6[-1])
        # feature1[-1] = feature1_4
        # feature2[-1] = feature2_4
        # feature3[-1] = feature3_4
        # feature4[-1] = feature4_4
        # feature5[-1] = feature5_4
        # feature6[-1] = feature6_4
        # CDDecoder
        feature_diff = []
        for i in range(len(feature_rgb)):
            feature_diff.append(self.CFEM[i](feature_rgb[i], feature_sar[i]))
        xc = self.CD_Decoder(feature_diff)
        bcd = self.seg(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifierCD(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.softmax(scd)
        bcd = torch.sigmoid(bcd)
        return scd.squeeze(1), bcd.squeeze(1)

from models.dual_vmamba import RGBXTransformer, vssm_tiny
class mmscd_sigma(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_sigma, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = vssm_tiny(depths=[2, 2, 9, 2])
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
        # self.MambaLayer = STM_MTGrootV3D_V2_DynamicEarth(640, False)
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/New_SECOND_dataset-BiGrootV3D_V1-STM_BiGrootV3D_V2-0114/resnet34/epoch66_Score38.41_mIOU72.90_Sek23.63_Fscd64.16_OA87.67.pth')
        # # 1. 过滤出预训练权重中在模型字典中也存在的键
        # for key, value in pretrained_weights.items():
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         # 检查新的键是否存在于模型的 state_dict 中
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/时空谱联合SCD/TSS-SCD/models/sigma/vssm_tiny_0230_ckpt_epoch_262.pth')
        # new_dict = pretrained_weights['model']
        # region_dict = self.backbone.state_dict()
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'layers.')):
        #         new_key = 'vssm.' + key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        # 防止权重不匹配
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         if key == 'patch_embed.conv1.weight':
        #             # 检查当前模型的 conv1.weight 形状
        #             current_conv1_weight = self.backbone.state_dict()[key]
        #             # 创建一个新的权重，形状与当前模型一致
        #             new_conv1_weight = torch.zeros_like(current_conv1_weight)
        #             # 将预训练权重的前3通道复制到新权重的前3通道
        #             new_conv1_weight[:, :3, :, :] = value
        #             # 将新权重添加到 updated_weights
        #             updated_weights[key] = new_conv1_weight
        #         else:
        #             if key in self.backbone.state_dict():
        #                 updated_weights[key] = value
        #     else:
        #         if key in self.backbone.state_dict():
        #             updated_weights[key] = value

        # self.backbone.load_state_dict(updated_weights, strict=False)
        # after_weight = self.backbone.state_dict()
        # print('Successfully loaded Backbone pre-training weights!')

        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD)

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

    # def bi_mamba_forward(self, x, y, z, d, s, b):
    #     xf_sm1, yf_sm1, zf_sm1, df_sm1, sf_sm1, bf_sm1 = self.MambaLayer(x, y, z, d, s, b)
    #     bf_sm2, sf_sm2, df_sm2, zf_sm2, yf_sm2, xf_sm2 = self.MambaLayer(b, s, d, z, y, x)
    #     x_f = xf_sm1 + xf_sm2
    #     y_f = yf_sm1 + yf_sm2
    #     z_f = zf_sm1 + zf_sm2
    #     d_f = df_sm1 + df_sm2
    #     s_f = sf_sm1 + sf_sm2
    #     b_f = bf_sm1 + bf_sm2
    #     return x_f, y_f, z_f, d_f, s_f, b_f


    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        feature_fuse = self.backbone.forward(rgb, sar)
        # 遍历A中的每个矩阵
        # feature1_4, feature2_4, feature3_4, feature4_4, feature5_4, feature6_4 = self.bi_mamba_forward(feature1[-1], feature2[-1], feature3[-1],feature4[-1], feature5[-1], feature6[-1])
        # feature1[-1] = feature1_4
        # feature2[-1] = feature2_4
        # feature3[-1] = feature3_4
        # feature4[-1] = feature4_4
        # feature5[-1] = feature5_4
        # feature6[-1] = feature6_4
        # CDDecoder
        xc = self.CD_Decoder(feature_fuse)
        bcd = self.seg(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifierCD(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.softmax(scd)
        bcd = torch.sigmoid(bcd)
        return scd.squeeze(1), bcd.squeeze(1)

# 用于SN6数据集
class mmscd_sigma_SN6(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_sigma_SN6, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = vssm_tiny(depths=[2, 2, 9, 2])
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
        # self.MambaLayer = STM_MTGrootV3D_V2_DynamicEarth(640, False)
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/checkpoints/SECOND/New_SECOND_dataset-BiGrootV3D_V1-STM_BiGrootV3D_V2-0114/resnet34/epoch66_Score38.41_mIOU72.90_Sek23.63_Fscd64.16_OA87.67.pth')
        # # 1. 过滤出预训练权重中在模型字典中也存在的键
        # for key, value in pretrained_weights.items():
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         # 检查新的键是否存在于模型的 state_dict 中
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        # pretrained_weights = torch.load('/root/autodl-fs/SGF-Net/models/sigma/vssmtiny_dp01_ckpt_epoch_292.pth')
        # new_dict = pretrained_weights['model']
        # region_dict = self.backbone.state_dict()
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'layers.')):
        #         new_key = 'vssm.' + key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        # 防止权重不匹配
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         if key == 'patch_embed.conv1.weight':
        #             # 检查当前模型的 conv1.weight 形状
        #             current_conv1_weight = self.backbone.state_dict()[key]
        #             # 创建一个新的权重，形状与当前模型一致
        #             new_conv1_weight = torch.zeros_like(current_conv1_weight)
        #             # 将预训练权重的前3通道复制到新权重的前3通道
        #             new_conv1_weight[:, :3, :, :] = value
        #             # 将新权重添加到 updated_weights
        #             updated_weights[key] = new_conv1_weight
        #         else:
        #             if key in self.backbone.state_dict():
        #                 updated_weights[key] = value
        #     else:
        #         if key in self.backbone.state_dict():
        #             updated_weights[key] = value

        # self.backbone.load_state_dict(updated_weights, strict=False)
        # after_weight = self.backbone.state_dict()
        # print('Successfully loaded Backbone pre-training weights!')

        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD)

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

    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        feature_fuse = self.backbone.forward(rgb, sar)
        # CDDecoder
        xc = self.CD_Decoder(feature_fuse)
        bcd = self.seg(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifierCD(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.softmax(scd)
        bcd = torch.sigmoid(bcd)
        return scd.squeeze(1), bcd.squeeze(1)

#  用于武汉数据集
class mmscd_sigma_wuhan(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_sigma_wuhan, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = vssm_tiny(depths=[2, 2, 9, 2])
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
        self.seg = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.seg, self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4)

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


    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        feature_fuse = self.backbone.forward(rgb, sar)
        xc = self.CD_Decoder(feature_fuse)
        bcd = self.seg(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        bcd = torch.sigmoid(bcd)
        return bcd.squeeze(1)

class mmscd_single_GCN(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_single_GCN, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.GCN = GCN(32,32)
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
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/时空谱联合SCD/TSS-SCD/checkpoints/BRIGHT/mmscd_single_fulldata_20250520_155609/best_model.pth')
        # 1. 过滤出预训练权重中在模型字典中也存在的键
        for key, value in pretrained_weights.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                # 检查新的键是否存在于模型的 state_dict 中
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        # new_dict = pretrained_weights['model']
        # print(pretrained_weights)
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         new_key = key
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        # 防止权重不匹配
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         if key == 'patch_embed.conv1.weight':
        #             # 检查当前模型的 conv1.weight 形状
        #             current_conv1_weight = self.backbone.state_dict()[key]
        #             # 创建一个新的权重，形状与当前模型一致
        #             new_conv1_weight = torch.zeros_like(current_conv1_weight)
        #             # 将预训练权重的前3通道复制到新权重的前3通道
        #             new_conv1_weight[:, :3, :, :] = value
        #             # 将新权重添加到 updated_weights
        #             updated_weights[key] = new_conv1_weight
        #         else:
        #             if key in self.backbone.state_dict():
        #                 updated_weights[key] = value
        #     else:
        #         if key in self.backbone.state_dict():
        #             updated_weights[key] = value

        self.backbone.load_state_dict(updated_weights, strict=False)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded HR Backbone pre-training weights!')

        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD)

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

    def extract_features(self, image):
        # image 形状: (batch, channels, height, width)
        batch_size, channels, h, w = image.shape
        features = image.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch, num_nodes, channels)
        return features

    def calculate_adjacency(self, feature, spatial_weight=0.5, feature_weight=0.5):
        batch_size, num_nodes, _ = feature.shape
        device = feature.device  # 获取设备信息

        grid_size = int(num_nodes ** 0.5)
        assert grid_size ** 2 == num_nodes, "特征节点数必须是完全平方数"
        h = w = grid_size

        adjacency = torch.zeros(batch_size, num_nodes, num_nodes, device=device)

        for b in range(batch_size):
            # 空间坐标网格
            y = torch.arange(h, device=device).view(-1, 1).repeat(1, w).view(-1)
            x = torch.arange(w, device=device).repeat(h)
            coord = torch.stack([x, y], dim=1).float()  # (num_nodes, 2)

            # 空间距离相似性
            spatial_dist = torch.cdist(coord, coord)  # (num_nodes, num_nodes)
            spatial_sim = 1 / (1 + spatial_dist)

            # 特征相似性（余弦相似度）
            feat_sim = F.cosine_similarity(
                feature[b].unsqueeze(1),  # (num_nodes, 1, channels)
                feature[b].unsqueeze(0),  # (1, num_nodes, channels)
                dim=2
            )
            # 综合相似性

            total_sim = spatial_weight * spatial_sim + feature_weight * feat_sim
            adjacency[b] = total_sim

        return adjacency
    def bi_mamba_forward(self, x, y):
        xf_sm1, yf_sm1 = self.MambaLayer(x, y)
        yf_sm2, xf_sm2 = self.MambaLayer(y, x)
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        return x_f, y_f


    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = rgb
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
        batch_size, channels, height, width = feature1_4.size()
        gcn_feature1_4 = self.extract_features(feature1_4)
        adjacency1_4 = self.calculate_adjacency(gcn_feature1_4)
        gcn_feature2_4 = self.extract_features(feature2_4)
        adjacency2_4 = self.calculate_adjacency(gcn_feature2_4)
        gcn_feature1_4 = self.GCN(adjacency1_4, gcn_feature1_4)
        gcn_feature2_4 = self.GCN(adjacency2_4, gcn_feature2_4)

        restored1_4 = gcn_feature1_4.permute(0, 2, 1)  # (4, 3, 65536)
        gcn_feature1_4 = restored1_4.view(batch_size, 640, height, width)  # (4, 3, 256, 256)
        restored2_4 = gcn_feature2_4.permute(0, 2, 1)  # (4, 3, 65536)
        gcn_feature2_4 = restored2_4.view(batch_size, 640, height, width)  # (4, 3, 256, 256)

        feature1[-1] = gcn_feature1_4
        feature2[-1] = gcn_feature2_4
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
class mmscd_siam_GCN(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
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

        self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3, self.CFEM_4]

        self.MambaLayer = STM_GrootV3D_V2(640, False)
        self.softmax = nn.Softmax(dim=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))
        updated_weights = {}
        # pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/时空谱联合SCD/TSS-SCD/checkpoints/BRIGHT/mmscd_single_fulldata_20250520_155609/best_model.pth')
        # # 1. 过滤出预训练权重中在模型字典中也存在的键
        # for key, value in pretrained_weights.items():
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         # 检查新的键是否存在于模型的 state_dict 中
        #         if new_key in self.backbone.state_dict():
        #             updated_weights[new_key] = value
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        new_dict = pretrained_weights['model']
        print(pretrained_weights)
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                new_key = key
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        # 防止权重不匹配
        # for key, value in new_dict.items():
        #     if key.startswith(('patch_embed.', 'levels.')):
        #         if key == 'patch_embed.conv1.weight':
        #             # 检查当前模型的 conv1.weight 形状
        #             current_conv1_weight = self.backbone.state_dict()[key]
        #             # 创建一个新的权重，形状与当前模型一致
        #             new_conv1_weight = torch.zeros_like(current_conv1_weight)
        #             # 将预训练权重的前3通道复制到新权重的前3通道
        #             new_conv1_weight[:, :3, :, :] = value
        #             # 将新权重添加到 updated_weights
        #             updated_weights[key] = new_conv1_weight
        #         else:
        #             if key in self.backbone.state_dict():
        #                 updated_weights[key] = value
        #     else:
        #         if key in self.backbone.state_dict():
        #             updated_weights[key] = value

        self.backbone.load_state_dict(updated_weights, strict=False)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded HR Backbone pre-training weights!')

        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD)

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


    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = rgb
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


class mmscd_siam_DynamicGCN(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_DynamicGCN, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.GCN1 = DynamicGCN(80)
        self.GCN2 = DynamicGCN(160)
        self.GCN3 = DynamicGCN(320)
        self.GCN4 = DynamicGCN(640)
        self.GCN5 = DynamicGCN(640)
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
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        new_dict = pretrained_weights['model']
        print(pretrained_weights)
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                new_key = key
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=False)
        print('Successfully loaded HR Backbone pre-training weights!')

        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD)

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


    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = rgb
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
            # gcn_f1 = f1.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f1 = self.GCN[i](f1)
            # maps_f1 = map_f1.contiguous().view(batch_size, self.channel_nums[i], *f1.size()[2:])

            # gcn_f2 = f2.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f2 = self.GCN[i](f2)
            # maps_f2 = map_f2.contiguous().view(batch_size, self.channel_nums[i], *f2.size()[2:])

            restored_f1 = f1 + map_f1
            restored_f2 = f2 + map_f2
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

# 用于武汉数据集（建筑物二值变化检测）
class mmscd_siam_DynamicGCN_wuhan(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_DynamicGCN_wuhan, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.GCN1 = DynamicGCN(80)
        self.GCN2 = DynamicGCN(160)
        self.GCN3 = DynamicGCN(320)
        self.GCN4 = DynamicGCN(640)
        self.GCN5 = DynamicGCN(640)
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
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        new_dict = pretrained_weights['model']
        print(pretrained_weights)
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                new_key = key
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value
        self.backbone.load_state_dict(updated_weights, strict=False)
        print('Successfully loaded HR Backbone pre-training weights!')

        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD)

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

    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = rgb
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
            map_f1 = self.GCN[i](f1)
            map_f2 = self.GCN[i](f2)
            restored_f1 = f1 + map_f1
            restored_f2 = f2 + map_f2
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
        bcd = torch.sigmoid(bcd)
        return bcd.squeeze(1)

class mmscd_siam_GCN_CL(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_CL, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.res_fea = self._make_layer(ResBlock, 336, 80, 6, stride=1)
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
            self.Seg_Decoder = Seg_Decoder_ResNet(self.channel_nums)
            # self.Seg_Decoder2 = Seg_Decoder_ResNet(self.channel_nums)
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
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 2, kernel_size=1))
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        new_dict = pretrained_weights['model']
        print(pretrained_weights)
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                new_key = key
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value

        self.backbone.load_state_dict(updated_weights, strict=False)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded HR Backbone pre-training weights!')

        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD)

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


    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = rgb
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

        rgb_fea = self.Seg_Decoder(feature1)
        sar_fea = self.Seg_Decoder(feature1)


        feature1_0 = self.res_fea(torch.cat([feature1[0], rgb_fea], dim=1))
        feature2_0 = self.res_fea(torch.cat([feature2[0], sar_fea], dim=1))

        feature1[0] = feature1_0
        feature2[0] = feature2_0


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
        # scd = self.softmax(scd)
        # bcd = torch.sigmoid(bcd)
        return rgb_fea, sar_fea, scd.squeeze(1), bcd.squeeze(1)



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


class mmscd_siam_GCN_concat(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_concat, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = GrootV_3D(depths=[2, 2, 9, 2])
        self.GCN1 = GCN(80, 80)
        self.GCN2 = GCN(160, 160)
        self.GCN3 = GCN(320, 320)
        self.GCN4 = GCN(640, 640)
        self.GCN5 = GCN(640, 640)
        self.GCN = [self.GCN1, self.GCN2, self.GCN3, self.GCN4, self.GCN5]
        if backbone == "resnet18" or backbone == "resnet34" or backbone == "GrootV":
            self.channel_num = [80, 160, 320, 640, 640]
            self.channel_nums = [80*2, 160*2, 320*2, 640*2, 640*2]
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
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 2, kernel_size=1))
        updated_weights = {}
        pretrained_weights = torch.load('/media/lenovo/课题研究/博士小论文数据/长时序变化检测/Long-term-SCD/CMSCD_lxg/grootv_cls_tiny.pth')
        new_dict = pretrained_weights['model']
        print(pretrained_weights)
        for key, value in new_dict.items():
            if key.startswith(('patch_embed.', 'levels.')):
                new_key = key
                if new_key in self.backbone.state_dict():
                    updated_weights[new_key] = value

        self.backbone.load_state_dict(updated_weights, strict=False)
        after_weight = self.backbone.state_dict()
        print('Successfully loaded HR Backbone pre-training weights!')

        for param in self.backbone.parameters():
            param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.classifierCD)

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


    def forward(self, rgb, sar):
        b, c, h, w = rgb.shape
        # features extraction from HR images
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = rgb
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

            gcn_f1 = f1.contiguous().view(batch_size, self.channel_num[i], -1)
            map_f1 = self.GCN[i](gcn_f1)
            maps_f1 = map_f1.contiguous().view(batch_size, self.channel_num[i], *f1.size()[2:])

            gcn_f2 = f2.contiguous().view(batch_size, self.channel_num[i], -1)
            map_f2 = self.GCN[i](gcn_f2)
            maps_f2 = map_f2.contiguous().view(batch_size, self.channel_num[i], *f2.size()[2:])

            restored_f1 = torch.cat([f1, maps_f1], dim=1)
            restored_f2 = torch.cat([f2, maps_f2], dim=1)
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
        # scd = self.softmax(scd)
        # bcd = torch.sigmoid(bcd)
        return scd.squeeze(1), bcd.squeeze(1)


# 采用小波变化增强浅层特征

class mmscd_siam_GCN_WT(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_WT, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.WT_opt = WTFMBlock(3,16)
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
            pretrained_weights = torch.load('/root/autodl-fs/SGF-Net/grootv_cls_tiny.pth')
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
        
# 最终模型（改变小波变化类型）
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
        # self.WT_sar = WTFMBlock(3, 16)
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

        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        # 加载预训练权重（关键修改）
        if pretrained:
            pretrained_weights = torch.load(
                '/root/autodl-fs/SGF-Net/best_model_val_65.0_test_64.6.pth')
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
        initialize_weights(self.CD_Decoder, self.seg, self.classifierCD)

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
        bcd = self.seg(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifierCD(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        return scd.squeeze(1), bcd.squeeze(1)


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
        self.WT_opt = WTFMBlock(3, 16)
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

        self.STMambaLayer1 = STM_GrootV3D_V2(640, False)
        self.STMambaLayer2 = STM_GrootV3D_V2(640, False)
        # self.STMambaLayer3 = STM_GrootV3D_V2(320, False)
        # self.STMambaLayer4 = STM_GrootV3D_V2(160, False)

        
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
            pretrained_weights = torch.load(
                '/root/autodl-fs/SGF-Net/checkpoints/SN6/mmscd_siam_GCN_WT_singleSTM_SN6_20251215_145831/best_model_val_85.6_test_85.6.pth')
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

    def load_adapted_pretrained_weights(self):
        """智能加载适配的预训练权重"""
        try:
            # 加载预训练权重
            pretrained_weights = torch.load('/root/autodl-fs/SGF-Net/grootv_cls_tiny.pth')
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

        xf_sm1, yf_sm1 = self.STMambaLayer1(feature1[-1], feature2[-1])
        yf_sm2, xf_sm2 = self.STMambaLayer1(feature2[-1], feature1[-1])
        x_f = xf_sm1 + xf_sm2
        y_f = yf_sm1 + yf_sm2
        feature1[-1] = x_f
        feature2[-1] = y_f

        xf_sm31, yf_sm31 = self.STMambaLayer1(feature1[-1], feature2[-1])
        yf_sm32, xf_sm32 = self.STMambaLayer1(feature2[-1], feature1[-1])
        x3_f = xf_sm31 + xf_sm32
        y3_f = yf_sm31 + yf_sm32
        feature1[-2] = x3_f
        feature2[-2] = y3_f

        
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


# 最终模型（改变小波变化类型）
class mmscd_siam_GCN_WT_singleSTM_1(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_WT_singleSTM, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.WT_opt = WTFMBlock(3, 16)
        # self.WT_sar = WTFMBlock(3, 16, wavelet_type='db4')
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

        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        # 加载预训练权重（关键修改）
        if pretrained:
            pretrained_weights = torch.load(
                '/root/autodl-fs/SGF-Net/best_model_val_65.0_test_64.6.pth')
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
        initialize_weights(self.CD_Decoder, self.seg, self.classifierCD)

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
        bcd = self.seg(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifierCD(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        return scd.squeeze(1), bcd.squeeze(1)


class mmscd_siam_GCN_WT_noSTM(nn.Module):
    """mmscd_siam_GCN_WT_singleSTM_1 without STMambaLayer - ablation study"""
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(mmscd_siam_GCN_WT_noSTM, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.WT_opt = WTFMBlock(3, 16)
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

        # No STMambaLayer
        self.softmax = nn.Softmax(dim=1)

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

        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 4, kernel_size=1))
        self.seg = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        if pretrained:
            pretrained_weights = torch.load(
                '/root/autodl-fs/SGF-Net/best_model_val_65.0_test_64.6.pth')
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
                    param.requires_grad = True
        else:
            self.load_adapted_pretrained_weights()
            for param in self.backbone.parameters():
                param.requires_grad = True
        initialize_weights(self.CD_Decoder, self.seg, self.classifierCD)

    def load_adapted_pretrained_weights(self):
        try:
            pretrained_weights = torch.load('/home/remote/Liyujie-daima/SGF-Net-master/grootv_cls_tiny.pth')
            new_dict = pretrained_weights['model']
            updated_weights = {}
            model_dict = self.backbone.state_dict()
            for key, value in new_dict.items():
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        updated_weights[key] = value
                    elif key == 'patch_embed.conv1.weight':
                        out_ch, in_ch, kh, kw = model_dict[key].shape
                        new_weight = torch.zeros(model_dict[key].shape)
                        repeat_factor = in_ch // value.size(1)
                        if repeat_factor > 0:
                            for i in range(repeat_factor):
                                start_ch = i * value.size(1)
                                end_ch = (i + 1) * value.size(1)
                                if end_ch <= in_ch:
                                    new_weight[:, start_ch:end_ch] = value / repeat_factor
                        updated_weights[key] = new_weight
                        print(f'Adapted input layer: {value.shape} -> {new_weight.shape}')
            self.backbone.load_state_dict(updated_weights, strict=False)
            print(f'Loaded {len(updated_weights)}/{len(model_dict)} pretrained layers')
        except Exception as e:
            print(f'Pretrained weight loading failed: {e}')

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
        xy_in = torch.empty(b, c, h, 2 * w).cuda()
        xy_in[:, :, :, 0:w] = opt
        xy_in[:, :, :, w:2*w] = sar
        feature_xy = self.backbone.forward(xy_in)

        feature1 = []
        feature2 = []
        for matrix in feature_xy:
            Bs, Cs, Hs, Ws = matrix.shape
            Ws = Ws // 2
            T1_part = matrix[:, :, :, 0:Ws]
            T2_part = matrix[:, :, :, Ws:2*Ws]
            feature1.append(T1_part)
            feature2.append(T2_part)

        feature1_region = feature1
        feature2_region = feature2

        # No STMambaLayer processing - features go directly to GCN

        feature1_c = []
        feature2_c = []
        for i in range(len(feature1)):
            f1 = feature1[i]
            f2 = feature2[i]
            batch_size, channels, height, width = f1.size()

            gcn_f1 = f1.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f1 = self.GCN[i](gcn_f1)
            maps_f1 = map_f1.contiguous().view(batch_size, self.channel_nums[i], *f1.size()[2:])

            gcn_f2 = f2.contiguous().view(batch_size, self.channel_nums[i], -1)
            map_f2 = self.GCN[i](gcn_f2)
            maps_f2 = map_f2.contiguous().view(batch_size, self.channel_nums[i], *f2.size()[2:])

            restored_f1 = self.fusions[i](feature1_region[i], maps_f1)
            restored_f2 = self.fusions[i](feature2_region[i], maps_f2)

            feature1_c.append(restored_f1)
            feature2_c.append(restored_f2)

        feature_diff = []
        for i in range(len(feature1_c)):
            feature_diff.append(self.CFEM[i](feature1_c[i], feature2_c[i]))

        xc = self.CD_Decoder(feature_diff)
        bcd = self.seg(xc)
        bcd = F.interpolate(bcd, size=(h, w), mode='bilinear', align_corners=False)
        scd = self.classifierCD(xc)
        scd = F.interpolate(scd, size=(h, w), mode='bilinear', align_corners=False)
        return scd.squeeze(1), bcd.squeeze(1)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MTGrootV3D_SV3(backbone='resnet34', pretrained=True, nclass=7, lightweight=True, M=6, Lambda=0.00005).to(device)
    # model = ST_VSSM_Siam().to(device)
    model = mmscd_siam_GCN_WT(backbone='GrootV', pretrained=False, nclass=4, lightweight=True, M=6, Lambda=0.00005).to(device)
    print(model)
    image1 = torch.randn(1, 3, 512, 512).to(device)
    MS1 = torch.randn(1, 1, 512, 512).to(device)
    MS1 = MS1.repeat(1, 3, 1, 1)
    scd, bcd = model(image1, MS1)
    # seg1, seg2, seg3, change = model(image1, image2, image3)
    # fs = model(image1, image2, image3, image4, image5, image6, MS1, MS2, MS3, MS4, MS5, MS6)
    # print(seg1)
    from thop import profile
    FLOPs, Params = profile(model, (image1, MS1))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))