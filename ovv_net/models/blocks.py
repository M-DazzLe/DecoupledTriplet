"""
OVV-Net 轻量级网络模块

核心组件：
1. DepthwiseSeparableConv - 深度可分离卷积（1/9计算量）
2. CoordinateAttention - 坐标注意力（捕捉长距离依赖）
3. PixelShuffleUpsample - 亚像素上采样（零参数、无棋盘格）
4. EncoderBlock/DecoderBlock - U-Net 基础模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 深度可分离卷积（Depthwise Separable Convolution）
# ============================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ============================================================
# 2. 坐标注意力（Coordinate Attention）
# ============================================================

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # [1]

#[Image of Cartesian coordinate system]
 #Coordinate Pooling
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


# ============================================================
# 3. 亚像素上采样（PixelShuffle Upsample）
# ============================================================

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        # PixelShuffle 需要通道数扩展 r^2 倍
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=1, stride=1, padding=0)
        self.ps = nn.PixelShuffle(scale_factor)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ============================================================
# 4. 编码器模块 (Encoder Block)
# ============================================================

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 两个深度可分离卷积，第一个负责下采样(如果stride>1)
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, stride=1)

        # 残差连接 (如果维度匹配)
        self.use_res = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_res:
            out = out + residual
        return out


# ============================================================
# 5. 解码器模块 (Decoder Block)
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 1. 上采样
        self.up = PixelShuffleUpsample(in_channels, in_channels // 2)

        # 2. 特征融合 (Concat后通道数 = in//2 + skip)
        self.merge_channels = in_channels // 2 + skip_channels

        # 3. 卷积融合
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(self.merge_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = self.up(x)

        # 处理可能得尺寸不匹配 (padding)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=True)

        out = torch.cat([x, skip], dim=1)
        out = self.conv(out)
        return out


# ============================================================
# 6. ASPP 模块 (空洞空间金字塔池化) - 专治超大团块漏诊
# ============================================================
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[2, 4, 8]):
        super(ASPP, self).__init__()
        modules = []
        # 1. 1x1 卷积 (保留局部高频细节，抓小结节)
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        # 2. 多个空洞卷积 (不同膨胀率，抓中等和大团块)
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))

        # 3. 全局平均池化 (获取图像级别的全局上下文信息)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        self.convs = nn.ModuleList(modules)

        # 4. 融合投影层
        self.project = nn.Sequential(
            nn.Conv2d((len(atrous_rates) + 2) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # 全局池化特征上采样融合
        pool_feat = self.global_pool(x)
        pool_feat = F.interpolate(pool_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(pool_feat)

        # 拼接所有感受野维度的特征并降维
        res = torch.cat(res, dim=1)
        return self.project(res)