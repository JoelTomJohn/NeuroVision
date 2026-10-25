# NeuroVision — Mentor Handover & Project Dossier

**Author / Developer:** Joel Tom John  
**Project:** NeuroVision — AI-Assisted Brain MRI Analysis & Quantitative Volumetrics  
**Handover Date:** October 25, 2026  
**Repository:** [https://github.com/JoelTomJohn/NeuroVision](https://github.com/JoelTomJohn/NeuroVision)

---

## 📌 Executive Summary

NeuroVision is an end-to-end, clinical research prototype designed for automated 3D multimodal brain tumor segmentation, quantitative morphometric analysis, spatial confidence visualization, and automated PDF reporting. 

The system is fully operational on Windows 10/11 and verified on Python 3.9.5. The deep learning backbone is powered by a verified **MONAI SegResNet** (3D residual encoder-decoder) checkpoint trained on the Brain Tumor Segmentation (BraTS) benchmark. All features, UI controls, analysis calculations, and automated tests are fully functional without external cloud dependencies.

---

## 🎯 Current Project State & Working Features

### 1. Multimodal 3D MRI Ingestion & Validation
- Simultaneously processes four co-registered diagnostic sequences: **T1c** (contrast-enhanced), **T1** (native), **T2** (T2-weighted), and **FLAIR**.
- Reads standard NIfTI formats (`.nii`, `.nii.gz`) using `nibabel`, extracting spatial affines and exact voxel spacing ($s_x, s_y, s_z$).
- Enforces strict dimension and resolution congruence checks across all four input channels.

### 2. Deep Learning Segmentation Engine (MONAI SegResNet)
- **Model Checkpoint:** `models/model.pt` (17.97 MB), cleanly stored within GitHub version control limits.
- **Inference Mode:** 3D sliding-window inference (`monai.inferers.sliding_window_inference`) with 50% Gaussian window overlap.
- **Hierarchical Consistency Enforcement:** Ensures clinical nesting invariants ($\text{ET} \subseteq \text{TC} \subseteq \text{WT}$).
- **Honest Execution:** Demo Mode is set to **OFF by default**. If inference fails on an invalid scan, it returns a descriptive error rather than silently faking results.

### 3. Quantitative Morphometrics & Analytics
- Computes exact tumor volume in **cubic centimeters ($\text{cm}^3$)** derived from NIfTI header voxel zooms.
- Computes physical bounding-box dimensions ($L \times W \times H$ in $\text{mm}$) along anatomical axes.
- Automatically calculates sub-region volume fractions (e.g. Enhancing Tumor as % of Whole Tumor).
- Computes spatial heuristic confidence metrics (mean, min, max, standard deviation) across tumor voxels.

### 4. Interactive Radiological Slice & 3D Viewer
- 2D axial slice viewer with standard radiological orientation (90° clockwise rotation; patient right on viewer left).
- Multi-mode display: Original MRI, AI Segmentation, Composite Overlay, and Side-by-Side comparison.
- **"🎯 Peak Tumor Slice"** button automatically snaps the slider to the axial slice with the largest tumor cross-section.
- Hardware-accelerated 3D point cloud rendered via Plotly `Scatter3d` with interactive rotation and multi-layer toggling.

### 5. Automated Clinical PDF Reporting
- Compiles downloadable PDF reports via ReportLab.
- Embeds metadata, quantitative volume tables, confidence metrics, a high-resolution axial key-slice thumbnail, and regulatory disclaimers.

### 6. One-Click Benchmark Demonstration & Evaluation
- Pre-bundled benchmark patient scan (`BraTS-GLI-00000-000`) enables instantaneous demonstration without requiring external uploads.
- Real-time Ground-Truth Benchmark Evaluation module computes Dice Similarity Coefficients against expert neuroradiologist annotations:
  - **Whole Tumor (WT):** **0.9182** (91.8%)
  - **Tumor Core (TC):** **0.9761** (97.6%)
  - **Enhancing Tumor (ET):** **0.9356** (93.6%)

---

## 📂 Key Files & Repository Map

```
NeuroVision/
├── BrainTumorAI/
│   ├── app.py                     # Main Streamlit application and multi-page routing
│   ├── requirements.txt           # Python dependencies pinned with minimum versions
│   ├── run_tests.py               # Test runner CLI (--fast, --all)
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # NIfTI loading, shape validation, zero-mean normalization
│   │   ├── inference.py           # MONAI SegResNet definition, sliding-window pipeline
│   │   ├── analysis.py            # Volumetric cm³, bounding box mm, Dice calculation
│   │   └── report.py              # ReportLab clinical PDF summary generator
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── components.py          # UI cards, badges, banners, and metric widgets
│   │   ├── styles.py              # Clinical CSS styling and color tokens
│   │   └── viewer.py              # 2D radiological slice viewer and 3D point cloud
│   ├── models/
│   │   ├── README.md              # Checkpoint specification, inputs/outputs, provenance
│   │   └── model.pt               # MONAI SegResNet pretrained weights (17.97 MB)
│   ├── data/
│   │   ├── .gitkeep
│   │   └── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
│   │       └── BraTS-GLI-00000-000/   # Bundled 5-file benchmark scan
│   ├── outputs/
│   │   └── README.md              # Runtime directory for generated PDF reports
│   └── tests/
│       ├── __init__.py
│       ├── test_units.py          # 12 automated unit tests (preproc, analysis, report, viewer)
│       └── test_benchmark.py      # Genuine SegResNet benchmark test vs ground-truth
├── docs/
│   ├── SETUP.md                   # Comprehensive Windows installation & launch instructions
│   ├── ARCHITECTURE.md            # System architecture, deep learning pipeline, data flow
│   ├── TESTING.md                 # Verification test results, actual logs, known boundaries
│   └── MENTOR_HANDOVER.md         # This document
├── .gitignore                     # Git exclusion rules (venv, secrets, cache, large datasets)
└── README.md                      # Primary project overview and quickstart guide
```

---

## ⚠️ Known Limitations & Responsible-Use Boundaries

1. **Research & Educational Prototype:** NeuroVision is designed strictly as an academic research prototype. It is **NOT an FDA-cleared or CE-marked medical device** and must never be used for autonomous clinical diagnosis or treatment planning.
2. **Preprocessing Requirements:** Scans must be co-registered and skull-stripped before submission. Raw unstripped DICOM acquisitions containing skull/facial bone may impair segmentation boundaries.
3. **Execution Runtime on CPU:** Sliding-window inference across a $240 \times 240 \times 155$ 4-channel volume on CPU requires approximately 45–90 seconds. CUDA GPU acceleration is supported if an NVIDIA card is available.
4. **Binary Scope:** The model segments gross tumor sub-regions but does not predict WHO histopathological grade, overall patient survival, or molecular biomarkers (e.g. IDH mutation or 1p/19q codeletion).

---

## 🔮 Recommended Next Steps & Future Research

1. **nnU-Net v2 Framework Integration:** Adapt the pipeline to support trained nnU-Net v2 checkpoints (`nnUNetTrainer__nnUNetPlans__3d_fullres`), leveraging self-configuring preprocessing and dynamic architectural topologies.
2. **DICOM PACS Ingestion:** Add native DICOM network ingestion (`pydicom`, `pynetdicom`, `SimpleITK`) to query and retrieve scans directly from clinical imaging archives.
3. **Calibrated Epistemic Uncertainty:** Incorporate Monte Carlo Dropout or deep ensembling to provide rigorous Bayesian confidence intervals rather than heuristic spatial probabilities.
4. **Longitudinal Tumor Tracking:** Implement automatic image co-registration between pre- and post-operative scans to quantify tumor volume response according to RANO (Response Assessment in Neuro-Oncology) criteria.

---

## 🛠️ Mentor Quickstart (3-Step Evaluation)

### Step 1: Open Terminal & Activate Environment
```powershell
cd NeuroVision\BrainTumorAI
.\venv\Scripts\Activate.ps1
```

### Step 2: Run Verification Test Suite
```powershell
python run_tests.py --fast
```
*(Confirms that all 12 core unit tests pass in under 2 seconds).*

### Step 3: Launch NeuroVision
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser:
1. Navigate to **"🔬 MRI Analysis"**.
2. Click **"⚡ Quick-Load BraTS Benchmark Scan"**.
3. Click **"🔬 Run AI Brain Tumor Segmentation"**.
4. Explore the interactive slices, 3D point cloud, and download the PDF report.
