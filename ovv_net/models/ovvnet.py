import torch
import torch.nn as nn
import torch.nn.functional as F
from ovv_net.models.blocks import DepthwiseSeparableConv, EncoderBlock, DecoderBlock, ASPP


class OVVNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config['in_channels']
        self.base_channels = config['base_channels']
        num_encoder_blocks = config.get('num_encoder_blocks', 4)
        num_decoder_blocks = config.get('num_decoder_blocks', 4)

        # === 1. Encoder ===
        self.encoder = nn.ModuleList()
        curr_channels = self.base_channels

        # Stem: 第一层卷积 (保持原尺寸)
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, curr_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(curr_channels),
            nn.ReLU(inplace=True)
        )

        enc_channels = [curr_channels]  # 记录跳跃连接的通道数

        for i in range(num_encoder_blocks):
            # 🔥 修复核心：
            # 1. 删除 use_softpool (blocks.py 不支持)
            # 2. 强制 stride=2 (必须下采样，否则和解码器不匹配)
            self.encoder.append(EncoderBlock(curr_channels, curr_channels * 2, stride=2))

            curr_channels *= 2
            enc_channels.append(curr_channels)

        # === 2. Bottleneck (最底层) ===
        self.bottleneck_conv = ASPP(curr_channels, curr_channels, atrous_rates=[2, 4, 8])

        # === 3. Decoder ===
        self.decoder = nn.ModuleList()
        for i in range(num_decoder_blocks):
            # Skip connection 来自 Encoder 的倒序
            # 假设 Encoder 下采样了 4 次，enc_channels 有 5 个元素 [Stem, E1, E2, E3, E4]
            # Bottleneck 接着 E4
            # Decoder 1 融合 E4 (enc_channels[-2])

            # 安全检查：防止索引越界
            skip_idx = -(i + 2)
            if abs(skip_idx) > len(enc_channels):
                raise ValueError(f"Decoder layer {i} cannot find matching skip connection!")

            skip_ch = enc_channels[skip_idx]
            out_ch = curr_channels // 2
            self.decoder.append(DecoderBlock(curr_channels, skip_ch, out_ch))
            curr_channels = out_ch

        # === 4. Deep Supervision Heads (深层监督) ===
        self.ds_head_1 = None
        self.ds_head_2 = None

        if num_decoder_blocks >= 2:
            self.ds_head_1 = nn.Conv2d(self.base_channels * 2, 1, kernel_size=1)
        if num_decoder_blocks >= 3:
            self.ds_head_2 = nn.Conv2d(self.base_channels * 4, 1, kernel_size=1)

        # === 5. Final Output ===
        self.final_conv = nn.Conv2d(curr_channels, 1, kernel_size=1)

    def forward(self, x):
        # 1. Stem
        x = self.stem(x)
        skips = [x]

        # 2. Encoder Path
        for block in self.encoder:
            x = block(x)  # EncoderBlock 内部已经处理了下采样
            skips.append(x)  # 将下采样后的特征存入 skip

        # 3. Bottleneck
        x = self.bottleneck_conv(x)

        # 4. Decoder Path
        decoder_outputs = []
        for i, block in enumerate(self.decoder):
            # 取对应的 Skip connection
            # skips 列表：[Stem(512), E1(256), E2(128), E3(64), E4(32)]
            # Bottleneck 输出的是 32
            # Decoder 1 上采样到 64，需要融合 E3 (skips[-2])
            skip_idx = -(i + 2)
            skip_feat = skips[skip_idx]

            x = block(x, skip_feat)
            decoder_outputs.append(x)

        # 5. Output
        final_out = self.final_conv(x)

        if self.training:
            aux_outputs = []
            # 辅助输出逻辑
            if self.ds_head_1 is not None and len(decoder_outputs) >= 2:
                feat_mid = decoder_outputs[-2]
                aux1 = self.ds_head_1(feat_mid)
                aux1 = F.interpolate(aux1, size=final_out.shape[2:], mode='bilinear', align_corners=False)
                aux_outputs.append(aux1)

            if self.ds_head_2 is not None and len(decoder_outputs) >= 3:
                feat_small = decoder_outputs[-3]
                aux2 = self.ds_head_2(feat_small)
                aux2 = F.interpolate(aux2, size=final_out.shape[2:], mode='bilinear', align_corners=False)
                aux_outputs.append(aux2)

            all_results = [final_out]
            all_results.extend(aux_outputs)
            return tuple(all_results)
        else:
            return final_out