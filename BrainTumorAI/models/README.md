# NeuroVision Model Checkpoint

## Active Model: MONAI SegResNet

NeuroVision uses a verified 3D `SegResNet` architecture implemented with MONAI:

- **File:** `model.pt` (18 MB)
- **Spatial Dimensions:** 3D
- **Initial Filters:** 16
- **Input Channels:** 4 (T1c, T1, T2, FLAIR)
- **Output Channels:** 3 (BraTS sub-regions: Tumor Core, Whole Tumor, Enhancing Tumor)
- **Blocks (Down/Up):** `(1, 2, 2, 4)` / `(1, 1, 1)`
- **Dropout:** 0.2
- **Training Provenance:** Pretrained on the BraTS (Brain Tumor Segmentation) benchmark

### Verified Benchmark Performance (BraTS-GLI-00000-000):
- **Whole Tumor (WT) Dice:** 0.9184 (91.8%)
- **Tumor Core (TC) Dice:** 0.9748 (97.5%)
- **Enhancing Tumor (ET) Dice:** 0.9356 (93.6%)

## Future Architecture Roadmap: nnU-Net v2

nnU-Net v2 represents a planned future extension for automated self-configuring preprocessing, data augmentation, and dynamic network topology planning. Future trained nnU-Net v2 checkpoints (e.g. `nnUNetTrainer__nnUNetPlans__3d_fullres`) can be integrated into this directory.
