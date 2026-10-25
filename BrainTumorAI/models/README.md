# NeuroVision Model Checkpoint Specification

## 🧠 Active Architecture: MONAI SegResNet (3D)

NeuroVision utilizes a verified 3D `SegResNet` architecture implemented via the **MONAI** (Medical Open Network for AI) framework:

- **Filename:** `model.pt`
- **Location:** `BrainTumorAI/models/model.pt`
- **File Size:** 18,840,620 bytes (17.97 MB)
- **SHA-256 Hash:** `860CCB3F1C21C99D0410AD8A1AC4EF6B8FAB60CEC0A503B0BA42675741A750AE`
- **Spatial Dimensions:** 3D $(X, Y, Z)$
- **Initial Filters:** 16
- **Input Channels:** 4 (T1c, T1, T2, FLAIR)
- **Output Channels:** 3 (Tumor Core, Whole Tumor, Enhancing Tumor)
- **Downsampling Blocks:** `(1, 2, 2, 4)` with residual skip connections
- **Upsampling Blocks:** `(1, 1, 1)`
- **Dropout Rate:** 0.2 across residual layers
- **Normalization:** Per-channel zero-mean, unit-variance across non-zero voxels

---

## 📥 Expected Model Inputs

The model expects a 4-dimensional floating-point NumPy array or PyTorch tensor of shape `(4, X, Y, Z)`:

| Channel Index | Modality Code | Diagnostic Sequence | Signal Characteristic |
| :---: | :---: | :--- | :--- |
| **0** | **T1c** | T1-weighted contrast-enhanced (Gadolinium) | Hyperintense breakdown of blood-brain barrier |
| **1** | **T1** | T1-weighted native non-contrast | Anatomical parenchymal reference |
| **2** | **T2** | T2-weighted spin-echo | Hyperintense free-water / fluid collections |
| **3** | **FLAIR** | T2 Fluid-Attenuated Inversion Recovery | Hyperintense peritumoral vasogenic edema |

Input dimensions must be skull-stripped and spatially co-registered. The bundled benchmark scan has spatial shape $(240, 240, 155)$.

---

## 📤 Expected Model Outputs

Inference is performed via `monai.inferers.sliding_window_inference` with sigmoid activation:

$$\hat{y} = \sigma(z) \in [0, 1]$$

| Output Channel | Clinical Region | Anatomical Composition | Binary Threshold |
| :---: | :--- | :--- | :---: |
| **0** | **Tumor Core (TC)** | Necrotic non-enhancing core + Enhancing tumor | $> 0.50$ |
| **1** | **Whole Tumor (WT)** | Peritumoral edema + Tumor Core | $> 0.50$ |
| **2** | **Enhancing Tumor (ET)** | Active hypervascular viable tumor rim | $> 0.50$ |

### Hierarchical Mask Consistency
Following sigmoid thresholding, topological consistency is strictly enforced:
- $\text{TC} = \text{TC} \cup \text{ET}$
- $\text{WT} = \text{WT} \cup \text{TC}$

Mutually disjoint anatomical labels are computed as:
- **NCR (Label 1):** Non-enhancing necrotic core ($\text{TC} \setminus \text{ET}$)
- **ED (Label 2):** Peritumoral vasogenic edema ($\text{WT} \setminus \text{TC}$)
- **ET (Label 3):** Enhancing tumor ($\text{ET}$)

---

## 🏆 Verified Benchmark Performance

Evaluated against expert neuroradiologist ground-truth masks on the bundled open benchmark subject (`BraTS-GLI-00000-000`):

| Region | Dice Similarity Coefficient (DSC) | Agreement Percentage |
| :--- | :---: | :---: |
| **Whole Tumor (WT)** | **0.9182** | **91.8%** |
| **Tumor Core (TC)** | **0.9761** | **97.6%** |
| **Enhancing Tumor (ET)** | **0.9356** | **93.6%** |

---

## 📜 Dataset & Weight Provenance

- **Training Benchmark:** The model weights were pretrained on the Brain Tumor Segmentation (BraTS) Challenge dataset (ASNR-MICCAI BraTS 2023 / 2021 cohort).
- **Citations:**
  - Baid, U., et al. "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification." *Frontiers in Neuroscience*, 15, 2021.
  - Bakas, S., et al. "Advancing The Cancer Genome Atlas glioma analysis with analytics, tissue images, and clinical data." *Scientific Data*, 4, 170117, 2017.
- **Licensing:**
  - MONAI software framework is licensed under **Apache 2.0**.
  - The model weights and bundled benchmark sample are provided strictly for academic, research, and non-commercial educational purposes.

---

## 📦 Checkpoint Acquisition & Version Control

Because `model.pt` is 17.97 MB, it is committed directly to the GitHub repository without requiring Git LFS or external cloud storage accounts.

If the file needs to be verified or restored:
```powershell
# Verify SHA256 integrity
Get-FileHash BrainTumorAI\models\model.pt
```

---

## 🔮 Roadmap: Future nnU-Net v2 Integration

The repository architecture is designed to support future nnU-Net v2 checkpoints:
- **Directory target:** `models/nnUNet/`
- **Expected format:** Standard nnU-Net v2 model directory containing `plans.json`, `dataset.json`, and fold checkpoints (`checkpoint_final.pth`).
- The current implementation maintains backward compatibility aliases in `backend/inference.py`:
  ```python
  run_nnunet_inference = run_segresnet_inference
  ```
