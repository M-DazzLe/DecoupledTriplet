"""LungAI 三阶段结节检测-诊断流水线 —— 可移植部署版（macOS / Apple Silicon MPS）。

自包含：模型代码、权重、脚本全部在本目录内，拷到任何 Mac 即可运行。
用法：
  1. 将病例 DICOM 文件夹放入 ./patients/（每例一个子文件夹）
  2. python run_timing.py
输出：./outputs/ 下每例的 txt 报告 + PNG 可视化 + NIfTI 3D 数据 + 计时汇总。
"""
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch
import scipy.ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

PKG_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG_ROOT))

from ovv_net.config.config import CONFIG
from ovv_net.models.ovvnet import OVVNet
from new_step3_model import HCE_ConvNeXt_3D

CKPT_S1 = str(PKG_ROOT / 'checkpoints' / 's1_ovvnet.pth')
CKPT_S2 = str(PKG_ROOT / 'checkpoints' / 's2_lightlungnet.pth')
CKPT_S3 = str(PKG_ROOT / 'checkpoints' / 's3_hcenet.pth')

PATIENTS_ROOT = PKG_ROOT / 'patients'
OUTPUT_DIR = PKG_ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_S1 = 16
BATCH_S3 = 16
USE_FP16 = False

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def sync():
    if DEVICE == 'mps':
        torch.mps.synchronize()
    elif DEVICE == 'cuda':
        torch.cuda.synchronize()


def amp_ctx():
    return torch.autocast(device_type='cuda', dtype=torch.float16) \
        if USE_FP16 and DEVICE == 'cuda' else nullcontext()


def load_medical_volume(path):
    print(f"⚙️ 正在加载患者数据: {path.name}")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(path))
    reader.SetFileNames(dicom_names)
    itk_img = reader.Execute()

    sz, sp = itk_img.GetSize(), itk_img.GetSpacing()
    new_sz = [int(sz[i] * sp[i]) for i in range(3)]
    rsp = sitk.ResampleImageFilter()
    rsp.SetOutputSpacing([1, 1, 1])
    rsp.SetSize(new_sz)
    rsp.SetDefaultPixelValue(-1000)
    rsp.SetOutputDirection(itk_img.GetDirection())
    rsp.SetOutputOrigin(itk_img.GetOrigin())
    itk_1mm = rsp.Execute(itk_img)

    vol = sitk.GetArrayFromImage(itk_1mm).astype(np.float32)
    norm_vol = (np.clip(vol, -1000, 400) + 1000) / 1400.0
    if norm_vol.shape[1] != 512 or norm_vol.shape[2] != 512:
        norm_vol = np.stack([cv2.resize(s, (512, 512)) for s in norm_vol])
    return norm_vol, vol


def get_3d_patch(vol, z, y, x, size=64):
    z_s, y_s, x_s = z - size // 2, y - size // 2, x - size // 2
    z_e, y_e, x_e = z_s + size, y_s + size, x_s + size
    pad_z = [max(0, -z_s), max(0, z_e - vol.shape[0])]
    pad_y = [max(0, -y_s), max(0, y_e - vol.shape[1])]
    pad_x = [max(0, -x_s), max(0, x_e - vol.shape[2])]
    crop = vol[max(0, z_s):z_e, max(0, y_s):y_e, max(0, x_s):x_e]
    patch = np.pad(crop, (pad_z, pad_y, pad_x), mode='constant', constant_values=0)
    return patch


def load_models():
    print("\n📦 正在组建专家会诊小组...")
    t0 = time.perf_counter()
    s1_net = OVVNet(CONFIG['MODEL_CONFIG']).to(DEVICE)
    s1_net.load_state_dict(torch.load(CKPT_S1, map_location=DEVICE)['state_dict'])
    s1_net.eval()

    from ovv_net.seg.train_mixed_seg8_final import LightLungNet
    s2_net = LightLungNet().to(DEVICE)
    s2_net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(CKPT_S2, map_location=DEVICE).items()})
    s2_net.eval()

    s3_net = HCE_ConvNeXt_3D(in_ch=1, base_c=32).to(DEVICE)
    s3_net.load_state_dict(torch.load(CKPT_S3, map_location=DEVICE)['state_dict'])
    s3_net.eval()
    sync()
    t_load = time.perf_counter() - t0

    with torch.no_grad(), amp_ctx():
        _ = s1_net(torch.zeros(1, 3, 512, 512, device=DEVICE))
        _ = s2_net(torch.zeros(1, 1, 512, 512, device=DEVICE))
        _ = s3_net(torch.zeros(1, 1, 64, 64, 64, device=DEVICE))
    sync()
    return s1_net, s2_net, s3_net, t_load


def run_joint_diagnosis(patient_dir, s1_net, s2_net, s3_net):
    patient_id = patient_dir.name
    timings = {}

    t0 = time.perf_counter()
    norm_vol, raw_vol = load_medical_volume(patient_dir)
    timings['preprocess'] = time.perf_counter() - t0
    depth = norm_vol.shape[0]

    all_s1_probs = np.zeros_like(norm_vol)
    t0 = time.perf_counter()
    with torch.no_grad(), amp_ctx():
        for start in tqdm(range(0, depth, BATCH_S1), desc="📡 [S1] 探测候选点"):
            end = min(start + BATCH_S1, depth)
            slabs = []
            for z in range(start, end):
                z_m, z_a = max(0, z - 1), min(depth - 1, z + 1)
                slabs.append(np.stack([norm_vol[z_m], norm_vol[z], norm_vol[z_a]]))
            batch = torch.from_numpy(np.stack(slabs)).float().to(DEVICE)
            out = torch.sigmoid(s1_net(batch))
            all_s1_probs[start:end] = out.squeeze(1).cpu().numpy()
    sync()
    timings['s1'] = time.perf_counter() - t0

    mask_s1 = all_s1_probs > 0.05
    labels, num = ndimage.label(mask_s1)
    candidates = []
    for obj in ndimage.find_objects(labels):
        z_c = int((obj[0].start + obj[0].stop) / 2)
        y_c = int((obj[1].start + obj[1].stop) / 2)
        x_c = int((obj[2].start + obj[2].stop) / 2)
        candidates.append({'z': z_c, 'y': y_c, 'x': x_c})

    print(f"🫁 [S2] 物理过滤中，原始候选点: {len(candidates)}")
    z_to_mask = {}
    t0 = time.perf_counter()
    with torch.no_grad(), amp_ctx():
        for cand in candidates:
            z = cand['z']
            if z not in z_to_mask:
                inp = torch.from_numpy(norm_vol[z]).unsqueeze(0).unsqueeze(0).to(DEVICE)
                p_mask = (torch.sigmoid(s2_net(inp)) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
                p_mask = ndi.binary_fill_holes(p_mask).astype(np.uint8)
                conts, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                v_conts = sorted(conts, key=cv2.contourArea, reverse=True)[:2]
                hull_mask = np.zeros_like(p_mask)
                for c in v_conts:
                    cv2.drawContours(hull_mask, [cv2.convexHull(c)], 0, 1, thickness=cv2.FILLED)
                z_to_mask[z] = hull_mask
    sync()
    timings['s2'] = time.perf_counter() - t0

    final_candidates = [c for c in candidates if z_to_mask[c['z']][c['y'], c['x']] > 0]
    print(f"✅ 过滤完毕，存活点: {len(final_candidates)}。开始三阶段深度会诊...")

    s3_results = []
    t0 = time.perf_counter()
    with torch.no_grad(), amp_ctx():
        for start in range(0, len(final_candidates), BATCH_S3):
            batch_cands = final_candidates[start:start + BATCH_S3]
            patches = np.stack([get_3d_patch(norm_vol, c['z'], c['y'], c['x'])
                                for c in batch_cands]).astype(np.float32)
            patch_tensor = torch.from_numpy(patches).unsqueeze(1).to(DEVICE)
            out = s3_net(patch_tensor)
            p_fpr = torch.sigmoid(out['fpr']).flatten().cpu().numpy()
            p_mal = torch.sigmoid(out['malignancy']).flatten().cpu().numpy()
            p_inv = torch.sigmoid(out['invasion']).flatten().cpu().numpy()
            p_mask = torch.sigmoid(out['mask']).cpu().numpy()
            for i, cand in enumerate(batch_cands):
                s3_results.append({'cand': cand, 'p_fpr': p_fpr[i], 'p_mal': p_mal[i],
                                   'p_inv': p_inv[i], 'p_mask': p_mask[i].squeeze()})
    sync()
    timings['s3'] = time.perf_counter() - t0
    timings['inference'] = timings['s1'] + timings['s2'] + timings['s3']

    t0 = time.perf_counter()
    report_path = OUTPUT_DIR / f"Diagnosis_{patient_id}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"🏥 患者 [{patient_id}] AI 联合会诊报告\n" + "=" * 40 + "\n")
        for i, res in enumerate(s3_results):
            cand = res['cand']
            p_fpr, p_mal, p_inv, p_mask = res['p_fpr'], res['p_mal'], res['p_inv'], res['p_mask']
            diag_str = f"候选点 #{i + 1} [Z:{cand['z']} Y:{cand['y']} X:{cand['x']}] -> "
            if p_fpr < 0.5:
                diag_str += "【判定：非结节 (假阳性拦截)】"
            else:
                diag_str += "【确认：真结节】"
                mal_type = "恶性 (Malignant)" if p_mal > 0.5 else "良性 (Benign)"
                diag_str += f" | 性质: {mal_type} (Prob:{p_mal:.2f})"
                if p_mal > 0.5:
                    inv_type = "存在浸润 (Invasion)" if p_inv > 0.5 else "原位/微浸润"
                    diag_str += f" | 浸润评价: {inv_type}"
            f.write(diag_str + "\n")

            patch_vol = get_3d_patch(norm_vol, cand['z'], cand['y'], cand['x'])
            mask_vol = (p_mask > 0.5).astype(np.uint8)
            itk_img = sitk.GetImageFromArray(patch_vol)
            itk_mask = sitk.GetImageFromArray(mask_vol)
            itk_img.SetSpacing([1.0, 1.0, 1.0])
            itk_mask.SetSpacing([1.0, 1.0, 1.0])
            nii_dir = OUTPUT_DIR / f"{patient_id}_3D_Data"
            nii_dir.mkdir(exist_ok=True)
            sitk.WriteImage(itk_img, str(nii_dir / f"Nodule_{i + 1}_CT.nii.gz"))
            sitk.WriteImage(itk_mask, str(nii_dir / f"Nodule_{i + 1}_Mask.nii.gz"))

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(norm_vol[cand['z']], cmap='gray')
            plt.plot(cand['x'], cand['y'], 'ro')
            plt.title("S1/S2 Detection")
            plt.subplot(1, 3, 2)
            plt.imshow(patch_vol[32], cmap='gray')
            plt.imshow(p_mask[32] > 0.5, alpha=0.4, cmap='Reds')
            plt.title("S3 HCE-Net 3D Mask")
            plt.subplot(1, 3, 3)
            results_text = f"FPR: {p_fpr:.2f}\nMAL: {p_mal:.2f}\nINV: {p_inv:.2f}"
            plt.text(0.1, 0.5, results_text, fontsize=15, color='red' if p_mal > 0.5 else 'green')
            plt.axis('off')
            plt.title("Diagnosis Metrics")
            plt.savefig(OUTPUT_DIR / f"{patient_id}_Nodule_{i + 1}.png")
            plt.close()
    timings['report'] = time.perf_counter() - t0
    print(f"\n🎉 诊断报告已生成: {report_path}（报告生成 {timings['report']:.2f}s）")
    return timings


def main():
    print(f"🖥️ 硬件: Apple Silicon (MPS) | torch {torch.__version__} | DEVICE={DEVICE}")
    patient_dirs = sorted([p for p in PATIENTS_ROOT.iterdir() if p.is_dir()])
    if not patient_dirs:
        print(f"未找到病例：请将 DICOM 病例文件夹放入 {PATIENTS_ROOT}")
        return
    s1_net, s2_net, s3_net, t_load = load_models()
    print(f"⏱️ 模型加载耗时: {t_load:.2f}s（一次，不计入单病例推理时间）\n")

    all_timings = []
    for patient_dir in patient_dirs:
        print(f"\n{'=' * 50}\n🧑 处理患者: {patient_dir.name}\n{'=' * 50}")
        t = run_joint_diagnosis(patient_dir, s1_net, s2_net, s3_net)
        t['patient'] = patient_dir.name
        all_timings.append(t)
        print(f"⏱️ 预处理 {t['preprocess']:.2f}s | S1 {t['s1']:.2f}s | S2 {t['s2']:.2f}s | "
              f"S3 {t['s3']:.2f}s | 推理合计 {t['inference']:.2f}s | 报告 {t['report']:.2f}s")

    if all_timings:
        infs = np.array([t['inference'] for t in all_timings])
        print(f"\n{'=' * 50}\n📊 汇总（{len(all_timings)} 例）\n{'=' * 50}")
        print(f"单病例推理时间: 平均 {infs.mean():.2f}s ± {infs.std():.2f}s "
              f"(min {infs.min():.2f}s, max {infs.max():.2f}s)")
        for t in all_timings:
            print(f"  {t['patient']}: 总 {t['inference']:.2f}s "
                  f"(预处理 {t['preprocess']:.2f} / S1 {t['s1']:.2f} / S2 {t['s2']:.2f} / S3 {t['s3']:.2f})")


if __name__ == "__main__":
    main()
