import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXt3DBlock(nn.Module):
    """ 3D 版 ConvNeXt 基础模块，极其强悍的特征提取器 """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)
        x = input + self.drop_path(x)
        return x


class MLPHead(nn.Module):
    """ 深度非线性逻辑中枢 """

    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class HCE_ConvNeXt_3D(nn.Module):
    def __init__(self, in_ch=1, base_c=32):
        super().__init__()

        # ================= 1. ENCODER (特征编码与下采样) =================
        # Stem 阶段：分为两步，提取原分辨率和 1/2 分辨率特征
        self.stem1 = nn.Sequential(
            nn.Conv3d(in_ch, base_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(base_c),
            nn.ReLU(inplace=True)
        )  # 输出: 64x64x64, base_c (用于 Skip 0)

        self.stem2 = nn.Conv3d(base_c, base_c, kernel_size=3, stride=2, padding=1)
        # 输出: 32x32x32, base_c

        self.stage1 = ConvNeXt3DBlock(base_c)  # 输出: 32x32x32, base_c (用于 Skip 1)
        self.down1 = nn.Conv3d(base_c, base_c * 2, kernel_size=2, stride=2)

        self.stage2 = ConvNeXt3DBlock(base_c * 2)  # 输出: 16x16x16, base_c*2 (用于 Skip 2)
        self.down2 = nn.Conv3d(base_c * 2, base_c * 4, kernel_size=2, stride=2)

        # 瓶颈层 Bottleneck
        self.stage3 = ConvNeXt3DBlock(base_c * 4)  # 输出: 8x8x8, base_c*4

        # ================= 2. DECODER (真正带 Addition Skip 的 3D U-Net) =================
        # Up 1: 8 -> 16
        self.up1 = nn.ConvTranspose3d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_c * 2, base_c * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_c * 2),
            nn.ReLU(inplace=True)
        )

        # Up 2: 16 -> 32
        self.up2 = nn.ConvTranspose3d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_c, base_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_c),
            nn.ReLU(inplace=True)
        )

        # Up 3: 32 -> 64
        self.up3 = nn.ConvTranspose3d(base_c, base_c, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(base_c, base_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_c, 1, kernel_size=1)  # 输出最终 Mask Logits
        )

        # ================= 3. CLASSIFICATION HEADS (级联逻辑) =================
        feat_dim = base_c * 4
        self.fpr_head = MLPHead(feat_dim, 1)  # 真假结节
        self.mal_head = MLPHead(feat_dim + 1, 1)  # 良恶性
        self.inv_head = MLPHead(feat_dim + 2, 1)  # 浸润度

    def forward(self, x):
        # ----------------- 编码流 (下山) -----------------
        x0 = self.stem1(x)  # [B, 32, 64, 64, 64] 💡 拦截：最高清边缘！
        x_stem2 = self.stem2(x0)  # [B, 32, 32, 32, 32]

        x1 = self.stage1(x_stem2)  # [B, 32, 32, 32, 32] 💡 拦截：次高清特征！
        x_down1 = self.down1(x1)  # [B, 64, 16, 16, 16]

        x2 = self.stage2(x_down1)  # [B, 64, 16, 16, 16] 💡 拦截：中层语义特征！
        x_down2 = self.down2(x2)  # [B, 128, 8, 8, 8]

        bottleneck = self.stage3(x_down2)  # [B, 128, 8, 8, 8] 纯纯的深层语义大脑

        # ----------------- 解码流 (上山 + 空间融合) -----------------
        # 第 1 级跳跃：融合 x2
        u1 = self.up1(bottleneck)  # 放大到 16x16x16
        u1 = u1 + x2  # ⚡ 核心：Addition Skip 2
        d1 = self.dec1(u1)

        # 第 2 级跳跃：融合 x1
        u2 = self.up2(d1)  # 放大到 32x32x32
        u2 = u2 + x1  # ⚡ 核心：Addition Skip 1
        d2 = self.dec2(u2)

        # 第 3 级跳跃：融合 x0 (绝杀！)
        u3 = self.up3(d2)  # 放大到 64x64x64
        u3 = u3 + x0  # ⚡ 核心：Addition Skip 0 (恢复极清毛刺边缘)
        mask_logits = self.dec3(u3)  # 最终 Mask [B, 1, 64, 64, 64]

        # ----------------- 掩码引导注意力 (MGA) -----------------
        mask_prob = torch.sigmoid(mask_logits)
        # 将超高清的 Mask 浓缩后作为门控，乘给瓶颈层
        mga_mask = F.interpolate(mask_prob, size=bottleneck.shape[2:], mode='trilinear', align_corners=False)
        gated_feat = bottleneck * mga_mask

        # ----------------- 条件级联分类 -----------------
        pooled_feat = F.adaptive_avg_pool3d(gated_feat, 1).flatten(1)

        fpr_logits = self.fpr_head(pooled_feat)
        fpr_prob = torch.sigmoid(fpr_logits)

        mal_input = torch.cat([pooled_feat, fpr_prob], dim=1)
        mal_logits = self.mal_head(mal_input)
        mal_prob = torch.sigmoid(mal_logits)

        inv_input = torch.cat([pooled_feat, fpr_prob, mal_prob], dim=1)
        inv_logits = self.inv_head(inv_input)

        return {
            'mask': mask_logits,  # 现在这个 Mask 精度极高
            'fpr': fpr_logits,
            'malignancy': mal_logits,
            'invasion': inv_logits
        }