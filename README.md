# Decoupled Triplet

Official inference code for **"Decoupled Triplet: A Lightweight Three-Stage Detection–Diagnosis Pipeline for Pulmonary Nodules"** (IEEE Transactions on Medical Imaging, under review).

Decoupled Triplet decomposes the complete pulmonary nodule workflow into three independently optimized stages:

- **S1 — OVV-Net** (10.43M): 2.5D recall-first candidate detection
- **S2 — LightLungNet** (0.23M): ERIS physical engine + lung-parenchyma filtering
- **S3 — HCE-Net** (0.82M): cascaded false-positive exclusion → malignancy → invasion diagnosis with 3D mask output

## Repository Contents

- `run_timing.py` — end-to-end inference pipeline with per-stage timing
- `ovv_net/` — model definitions for S1 (OVV-Net) and S2 (LightLungNet)
- `new_step3_model.py` — model definition for S3 (HCE-Net)
- `requirements.txt` — Python dependencies

## Quick Start

```bash
conda create -n decoupled_triplet python=3.10 -y
conda activate decoupled_triplet
pip install -r requirements.txt
```

1. Place patient DICOM folders into `patients/` (one subfolder per patient, containing that patient's `.dcm` series).
2. Place the three stage checkpoints into `checkpoints/` (see below).
3. Run `python run_timing.py`.

Outputs are written to `outputs/`: per-candidate diagnosis reports (`Diagnosis_*.txt`), visualization panels (`*_Nodule_*.png`), and 64³ CT-patch/mask pairs in NIfTI (`*_3D_Data/`), plus per-stage timing.

## Model Checkpoints

The three stage checkpoints (`s1_ovvnet.pth`, `s2_lightlungnet.pth`, `s3_hcenet.pth`, ~136 MB total) **will be released together with the training code upon formal publication** of the manuscript.

## Training Code and Data

The training code, the 630-CT clinical dataset, and the self-training protocol will be released upon acceptance. The clinical data is patient-privacy-restricted and will be shared under a data-use agreement.

## Hardware Notes

The pipeline runs on ordinary computers: 6.25 ± 1.43 s per scan on an NVIDIA GeForce RTX 4060 Laptop GPU, and 20.84 ± 6.86 s on an Apple M5 Mac (26 cases × 3 rounds; see the paper for details). Apple Silicon requires torch with MPS backend (included in the standard macOS pip wheel).

## Citation

If you use this code, please cite our paper (citation details will be added upon publication).
