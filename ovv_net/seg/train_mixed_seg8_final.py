import os
import glob
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import logging
import datetime

# ================= 🔧 配置区域 (服务器版) =================
LUNA_ROOT = "/home/stu1/lung_ovv/lung_data/luna_processed_dual"
COVID_ROOT = "/home/stu1/lung_ovv/lung_data/covid_processed_dual"
# macOS 根文件系统只读，无法创建 /home/stu1；该目录仅训练时使用，
# 推理时无功能作用，Mac 部署时指向 /tmp 即可
if platform.system() == "Darwin":
    CHECKPOINT_DIR = "/tmp/lung_mixed_checkpoints"
else:
    CHECKPOINT_DIR = "/home/stu1/lung_ovv/outputs/checkpoints/lung_mixed"
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 50
NUM_WORKERS = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ===============================================

def get_logger(log_dir):
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_file = os.path.join(log_dir, f"training_lung_mixed_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.log")
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# === 1. 数据集定义 (🔥 核心修改：动态剥离旧通道) ===
class UniversalLungDataset(Dataset):
    def __init__(self, root_dir, repeat=1):
        self.img_dir = os.path.join(root_dir, "imgs")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.files = sorted(glob.glob(os.path.join(self.img_dir, "*.npy")))
        self.repeat = repeat

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        actual_idx = idx % len(self.files)
        img_path = self.files[actual_idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))

        img = np.load(img_path).astype(np.float32)  # 原本是 [2, 512, 512]
        mask = np.load(mask_path).astype(np.float32)

        # 🌟 偷梁换柱：我们只取通道 0 (原图 CT)，扔掉你之前用 CPU 跑出来的慢速 DoG 通道！
        # 提取后 raw_img 形状变为 [1, 512, 512]
        raw_img = img[0:1, :, :]

        return torch.from_numpy(raw_img), torch.from_numpy(mask).unsqueeze(0)


# === 2. 模型定义 (🔥 引入纯 GPU 物理阶梯特征提取) ===

class MedicalERISChannel(nn.Module):
    """
    源自底层优化逻辑的纯 GPU 物理阶梯特征提取器。
    利用 Tensor 的 Broadcast 机制实现 $O(1)$ 时间复杂度的多阈值区间分割。
    """

    def __init__(self, segments=8):
        super().__init__()
        self.segments = segments
        # CT 归一化后是 0~1，将其死死钉入 8 个固定的物理阶梯
        steps = torch.linspace(0, 1.0, segments + 1)
        self.register_buffer('lower_bounds', steps[:-1].view(1, segments, 1, 1))
        self.register_buffer('upper_bounds', steps[1:].view(1, segments, 1, 1))
        # 赋予灰度权重，合成单通道特征图
        weights = torch.linspace(1.0 / segments, 1.0, segments)
        self.register_buffer('step_weights', weights.view(1, segments, 1, 1))

    def forward(self, x):
        # 瞬间生成 8 张二值化 Mask [Batch, 8, H, W]
        masks = ((x >= self.lower_bounds) & (x < self.upper_bounds)).float()
        # 降维拍扁为 1 张高对比度的阶梯特征图 [Batch, 1, H, W]
        eris_channel = torch.sum(masks * self.step_weights, dim=1, keepdim=True)
        return eris_channel


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class LightLungNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_c = 64
        # 🌟 内置 ERIS 引擎
        self.eris_extractor = MedicalERISChannel(segments=8)

        # 首层卷积，输入依然是 2 个通道 (原图 + GPU动态生成的ERIS图)
        self.c1 = nn.Sequential(nn.Conv2d(2, base_c, 3, padding=1, bias=False), nn.BatchNorm2d(base_c),
                                nn.ReLU(inplace=True), DepthwiseSeparableConv(base_c, base_c))
        self.p1 = nn.MaxPool2d(2)
        self.c2 = nn.Sequential(DepthwiseSeparableConv(base_c, base_c * 2),
                                DepthwiseSeparableConv(base_c * 2, base_c * 2))
        self.p2 = nn.MaxPool2d(2)
        self.b = nn.Sequential(DepthwiseSeparableConv(base_c * 2, base_c * 4),
                               DepthwiseSeparableConv(base_c * 4, base_c * 4))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c3 = nn.Sequential(DepthwiseSeparableConv(base_c * 6, base_c * 2),
                                DepthwiseSeparableConv(base_c * 2, base_c * 2))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c4 = nn.Sequential(DepthwiseSeparableConv(base_c * 3, base_c), DepthwiseSeparableConv(base_c, base_c))
        self.out = nn.Conv2d(base_c, 1, 1)

    def forward(self, x):
        # 🌟 x 输入时的形状是 [Batch, 1, H, W]
        with torch.no_grad():
            eris_map = self.eris_extractor(x)

        # 🌟 在这里完成真正的双通道拼接！
        fused_x = torch.cat([x, eris_map], dim=1)  # 变成 [Batch, 2, H, W]

        c1 = self.c1(fused_x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        b = self.b(p2)
        u1 = self.up1(b)
        if u1.shape != c2.shape: u1 = nn.functional.interpolate(u1, size=c2.shape[2:])
        c3 = self.c3(torch.cat([u1, c2], dim=1))
        u2 = self.up2(c3)
        if u2.shape != c1.shape: u2 = nn.functional.interpolate(u2, size=c1.shape[2:])
        c4 = self.c4(torch.cat([u2, c1], dim=1))
        return self.out(c4)


# === 3. 损失函数 ===
class DiceLoss(nn.Module):
    def forward(self, pred, target, smooth=1e-5):
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()


class MorphologicalBoundaryLoss(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def extract_boundary(self, x):
        dilation = F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        erosion = -F.max_pool2d(-x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        return dilation - erosion

    def forward(self, pred, target):
        pred_boundary = self.extract_boundary(pred)
        target_boundary = self.extract_boundary(target)
        return F.mse_loss(pred_boundary, target_boundary)


# === 4. 训练主循环 ===
def main():
    logger = get_logger(CHECKPOINT_DIR)
    logger.info("🚀 正在装载数据 (启动在轨 GPU 特征提取模式)...")

    ds_luna = UniversalLungDataset(LUNA_ROOT, repeat=1)
    ds_covid = UniversalLungDataset(COVID_ROOT, repeat=10)
    mixed_dataset = ConcatDataset([ds_luna, ds_covid])
    logger.info(f"📊 混合数据集总规模: {len(mixed_dataset)} 张切片")

    loader = DataLoader(mixed_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)

    model = LightLungNet().to(DEVICE)

    if torch.cuda.device_count() > 1:
        logger.info(f"🔥 检测到 {torch.cuda.device_count()} 张 GPU，启动 DataParallel 双卡并行训练！")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    criterion_boundary = MorphologicalBoundaryLoss(kernel_size=5)

    best_loss = float('inf')

    logger.info("🔥 开始形变感知混合训练 (ERIS Dual-Stream + Boundary-Aware + AMP + Multi-GPU)...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_bce, epoch_dice, epoch_bound, epoch_total = 0, 0, 0, 0
        alpha = min(1.0, epoch / 5.0)

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Alpha={alpha:.2f}]",
                    mininterval=30.0, ncols=100, ascii=True, leave=False)

        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                raw_logits = model(imgs)
                l_bce = criterion_bce(raw_logits, masks)

                preds_prob = torch.sigmoid(raw_logits)
                l_dice = criterion_dice(preds_prob, masks)
                l_bound = criterion_boundary(preds_prob, masks)

                loss = 0.5 * l_bce + 0.5 * l_dice + alpha * 1.0 * l_bound

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_bce += l_bce.item()
            epoch_dice += l_dice.item()
            epoch_bound += l_bound.item()
            epoch_total += loss.item()

            pbar.set_postfix({'Total': f"{loss.item():.4f}", 'Bound': f"{l_bound.item():.4f}"})

        avg_total = epoch_total / len(loader)
        avg_bce = epoch_bce / len(loader)
        avg_dice = epoch_dice / len(loader)
        avg_bound = epoch_bound / len(loader)

        log_msg = (f"✅ Epoch {epoch + 1}/{EPOCHS} Summary | Total Loss: {avg_total:.4f} | "
                   f"BCE: {avg_bce:.4f} | Dice: {avg_dice:.4f} | Bound: {avg_bound:.4f} | Alpha: {alpha:.2f}")
        logger.info(log_msg)

        if avg_total < best_loss:
            best_loss = avg_total
            save_path = os.path.join(CHECKPOINT_DIR, "best_mixed_eris.pth")

            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)

            logger.info(f"⭐ 已保存新一代 ERIS Best Model! -> {save_path}")


if __name__ == "__main__":
    main()