import os
from pathlib import Path

# ================= 路径配置 =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# 请确保这里指向你的数据路径
DATA_ROOT = Path('/home/stu1/lung_ovv/lung_data/luna16')
PREPROCESS_DIR = DATA_ROOT / 'raw_clean_data'

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
LOG_DIR = OUTPUT_DIR / 'logs'
RESULT_DIR = OUTPUT_DIR / 'results'
SPLIT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'splits'

for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, RESULT_DIR, SPLIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ================= 训练配置 =================
TRAIN_SUBSETS = [0, 1, 2, 3, 4, 5, 6, 10]
VAL_SUBSETS = [7, 8]
TEST_SUBSETS = [9]

# 🔥 核心修改 1: 全图尺寸
# 我们不再做裁剪，而是输入 512x512 的整张切片
CROP_SIZE = (512, 512)

# 🔥 核心修改 2: 显存保护
# 512x512 的计算量很大，16G 显存通常只能跑 Batch=2 或 4
BATCH_SIZE = 4

# 增强参数
AUG_ROTATION_RANGE = 10
AUG_FLIP_PROB = 0.5
AUG_NOISE_STD = 0.005

# 训练超参
NUM_WORKERS = 4
PIN_MEMORY = True
# 全图训练因为样本（切片）数量巨大，通常不需要跑那么多 epoch
EPOCHS = 400
PATIENCE = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# 调度器
LR_SCHEDULER = 'ReduceLROnPlateau'
LR_FACTOR = 0.5
LR_PATIENCE = 10
LR_MIN = 1e-6

# Loss (保持不变，分割 Loss 天然支持检测任务)
TVERSKY_ALPHA = 0.15
TVERSKY_BETA = 0.85
TVERSKY_WEIGHT = 0.7
BCE_WEIGHT = 0.3

# 模型配置
MODEL_CONFIG = {
    'in_channels': 3,      # z-1, z, z+1 三层作为 3 通道输入
    'base_channels': 32,   # 基础通道数
    'num_encoder_blocks': 4, # 🔥 增加深度：512尺寸建议下采样 4 次 (512->32)，捕捉更大感受野
    'num_decoder_blocks': 4,
}

DEVICE = 'cuda'

CONFIG = {
    'PROJECT_ROOT': PROJECT_ROOT,
    'PREPROCESS_DIR': PREPROCESS_DIR,
    'OUTPUT_DIR': OUTPUT_DIR,
    'CHECKPOINT_DIR': CHECKPOINT_DIR,
    'LOG_DIR': LOG_DIR,
    'TRAIN_SUBSETS': TRAIN_SUBSETS,
    'VAL_SUBSETS': VAL_SUBSETS,
    'INPUT_SIZE': CROP_SIZE,
    'BATCH_SIZE': BATCH_SIZE,
    'NUM_WORKERS': NUM_WORKERS,
    'PIN_MEMORY': PIN_MEMORY,
    'NUM_EPOCHS': EPOCHS,
    'EARLY_STOP_PATIENCE': PATIENCE,
    'LEARNING_RATE': LEARNING_RATE,
    'WEIGHT_DECAY': WEIGHT_DECAY,
    'LR_SCHEDULER': LR_SCHEDULER,
    'LR_FACTOR': LR_FACTOR,
    'LR_PATIENCE': LR_PATIENCE,
    'LR_MIN': LR_MIN,
    'TVERSKY_ALPHA': TVERSKY_ALPHA,
    'TVERSKY_BETA': TVERSKY_BETA,
    'TVERSKY_WEIGHT': TVERSKY_WEIGHT,
    'BCE_WEIGHT': BCE_WEIGHT,
    'MODEL_CONFIG': MODEL_CONFIG,
    'DEVICE': DEVICE,
}