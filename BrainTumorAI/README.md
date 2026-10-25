# 🧠 NeuroVision — AI-Assisted Brain MRI Analysis & Quantitative Volumetrics

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework-Streamlit](https://img.shields.io/badge/frontend-Streamlit%201.50-red.svg)](https://streamlit.io/)
[![Deep Learning-MONAI](https://img.shields.io/badge/deep--learning-MONAI%20SegResNet-brightgreen.svg)](https://monai.io/)
[![License-Research](https://img.shields.io/badge/license-Academic%20Research-lightgrey.svg)](LICENSE)
[![Status-Verified](https://img.shields.io/badge/status-Benchmarked%20%26%20Tested-success.svg)](docs/TESTING.md)

**NeuroVision** is a clinical research prototype for multimodal 3D brain magnetic resonance imaging (MRI) segmentation, quantitative volumetric analysis, confidence quantification, and standardized reporting. 

Powered by a deep learning **MONAI SegResNet** (3D residual encoder-decoder) architecture pretrained on the Brain Tumor Segmentation (BraTS) benchmark, NeuroVision automatically delineates whole tumor, tumor core, and active contrast-enhancing sub-compartments with verified high Dice similarity coefficients.

---

## 📚 Table of Contents

- [Key Features](#-key-features)
- [Deep Learning Architecture](#-deep-learning-architecture)
- [Verified Benchmark Performance](#-verified-benchmark-performance)
- [Required MRI Modalities](#-required-mri-modalities)
- [Repository Structure](#-repository-structure)
- [Quickstart & Windows Launch](#-quickstart--windows-launch)
- [Demonstration & Usage Workflow](#-demonstration--usage-workflow)
- [Automated Testing & Verification](#-automated-testing--verification)
- [Documentation Index](#-documentation-index)
- [Limitations & Responsible-Use Notice](#-limitations--responsible-use-notice)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [Provenance & Citations](#-provenance--citations)

---

## 🌟 Key Features

1. **Multimodal 3D MRI Segmentation:**
   - Simultaneous processing of four co-registered diagnostic MRI sequences: **T1c** (contrast-enhanced), **T1** (native), **T2** (T2-weighted), and **FLAIR** (fluid-attenuated inversion recovery).
   - Delineation of standard BraTS clinical sub-regions:
     - **Whole Tumor (WT):** Peritumoral vasogenic edema + non-enhancing core + active enhancing tumor.
     - **Tumor Core (TC):** Necrotic / cystic core + active enhancing tumor.
     - **Enhancing Tumor (ET):** Active contrast-enhancing hypervascular rim and nodular core.

2. **Interactive Radiological Viewer:**
   - Multi-mode axial visualization: *Original MRI*, *AI Segmentation*, *Radiological Overlay*, and *Side-by-Side Comparison*.
   - Multi-region overlay toggling with distinct clinical color swatches.
   - Synchronized slice navigation slider, previous/next controls, and an instant **"🎯 Peak Tumor Slice"** jump button.

3. **Quantitative Volumetric & Morphometric Analysis:**
   - Calculation of tumor volume in **cubic centimeters ($\text{cm}^3$)** derived from exact NIfTI voxel spacing metadata ($s_x \times s_y \times s_z$).
   - Physical orthogonal bounding-box dimensions ($L \times W \times H$ in $\text{mm}$) along anatomical planes.
   - Sub-compartment volume fractions (Enhancing Tumor % and Tumor Core % relative to Whole Tumor).

4. **Honest Confidence & Explainability:**
   - Spatial voxel-wise probability heatmap rendered via Plotly.
   - Summary statistics (Mean, Min, Max, and Standard Deviation) across predicted tumor voxels.
   - Transparent disclosure distinguishing spatial heuristic probability activations from Bayesian calibrated posterior uncertainty.

5. **Interactive 3D Volumetric Point Cloud:**
   - Hardware-accelerated 3D point cloud rendered via Plotly `Scatter3d`.
   - Multi-layer spatial separation with orbital rotation, panning, and zoom controls.

6. **Standardized Clinical PDF Reporting:**
   - One-click compilation of downloadable clinical summary reports using ReportLab.
   - Complete scan metadata, volumetric findings, confidence metrics, embedded key axial slice thumbnail, and regulatory disclaimers.

7. **One-Click Benchmark Demonstration:**
   - Pre-bundled benchmark patient scan (`BraTS-GLI-00000-000`) for instantaneous testing without external file uploads.
   - Ground-truth evaluation module calculating real Dice Similarity Coefficients against expert neuroradiologist annotations.

---

## 🏗️ Deep Learning Architecture

NeuroVision deploys a 3D residual encoder-decoder network from the **MONAI** framework:

- **Architecture:** `SegResNet` (3D residual encoder-decoder)
- **Framework:** PyTorch & MONAI
- **Input Channels:** 4 (Channel 0: T1c, Channel 1: T1, Channel 2: T2, Channel 3: FLAIR)
- **Output Channels:** 3 (Sigmoid activations for Tumor Core, Whole Tumor, Enhancing Tumor)
- **Base Filters:** 16 initial convolution filters
- **Encoder Blocks:** `(1, 2, 2, 4)` with residual skip connections and 0.2 dropout
- **Decoder Blocks:** `(1, 1, 1)` with upsampling convolutions
- **Pretrained Weights:** `models/model.pt` (17.97 MB)
- **Inference Strategy:** 3D sliding-window inference (`roi_size=(240, 240, 160)`, `overlap=0.5`, Gaussian importance weighting)
- **Hardware Optimization:** Verified on CPU execution (~45–90s); automatically leverages NVIDIA CUDA GPU acceleration when available.

---

## 🏆 Verified Benchmark Performance

Evaluated against expert neuroradiologist ground truth on benchmark scan `BraTS-GLI-00000-000`:

| Sub-Region | Clinical Definition | Predicted Voxels | Ground-Truth Voxels | Dice Similarity Coefficient (DSC) | Agreement |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **WT** | Whole Tumor (Edema + Core + Enhancing) | 63,522 | 57,305 | **0.9182** | **91.8%** |
| **TC** | Tumor Core (Necrotic + Enhancing) | 43,546 | 44,469 | **0.9761** | **97.6%** |
| **ET** | Enhancing Tumor (Active rim) | 30,473 | 32,731 | **0.9356** | **93.6%** |

---

## 📋 Required MRI Modalities

NeuroVision requires four co-registered, skull-stripped 3D MRI volumes in NIfTI format (`.nii` or `.nii.gz`):

| Sequence | Clinical Role | Signal Characteristic |
| :--- | :--- | :--- |
| **T1c** | Delineates active breakdown of blood-brain barrier | Hyperintense contrast enhancement |
| **T1** | Anatomical reference sequence | Gray/white matter parenchymal boundary |
| **T2** | Highlights abnormal tissue free-water content | Hyperintense edema & fluid collection |
| **FLAIR** | Suppresses normal cerebrospinal fluid (CSF) signal | Hyperintense peritumoral vasogenic edema |

*Note: Ground-truth segmentation masks (e.g. `*-seg.nii.gz`) must NEVER be uploaded as one of the four MRI input channels.*

---

## 📂 Repository Structure

```
NeuroVision/
├── BrainTumorAI/
│   ├── app.py                     # Main Streamlit application and multi-page routing
│   ├── requirements.txt           # Python dependencies pinned with minimum versions
│   ├── run_tests.py               # Test runner CLI (--fast, --all)
│   ├── backend/                   # Preprocessing, MONAI inference, analysis, reporting
│   ├── frontend/                  # Viewer, UI components, styling
│   ├── models/                    # Model checkpoint (model.pt) and specification
│   ├── data/                      # Bundled BraTS benchmark sample (BraTS-GLI-00000-000)
│   ├── outputs/                   # Runtime output directory for generated reports
│   └── tests/                     # Unit and benchmark test suites
├── docs/
│   ├── SETUP.md                   # Windows setup & installation guide
│   ├── ARCHITECTURE.md            # Technical architecture and data flow specification
│   ├── TESTING.md                 # Complete verification test results and logs
│   └── MENTOR_HANDOVER.md         # Mentor handover dossier and roadmap
├── .gitignore                     # Git exclusion rules
└── README.md                      # Primary project readme
```

---

## 🚀 Quickstart & Windows Launch

### 1. Prerequisites
- Windows 10 or 11 (64-bit)
- Python 3.9.0 to 3.9.13 (Verified on **Python 3.9.5**)
- 8 GB RAM minimum (16 GB recommended)

### 2. Setup & Virtual Environment
Open PowerShell:

```powershell
# Navigate to project directory
cd NeuroVision\BrainTumorAI

# Create virtual environment (if not already present)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Launch NeuroVision
```powershell
streamlit run app.py
```

*Or launch directly via Python module:*
```powershell
python -m streamlit run app.py
```

The web application will open automatically at `http://localhost:8501`.

For detailed Windows setup assistance, see [docs/SETUP.md](file:///c:/Users/thoma/Downloads/BrainTumorAI/docs/SETUP.md).

---

## 🖥️ Demonstration & Usage Workflow

1. **Dashboard:**
   - Launch app; inspect system status badge, 4 modality specification cards, and features.
   - Click **"🚀 Start New Analysis"** or select **"🔬 MRI Analysis"** in the sidebar.
2. **Load Benchmark Sample:**
   - Click **"⚡ Quick-Load BraTS Benchmark Scan"** to load `BraTS-GLI-00000-000` instantaneously.
   - Dimensions `(4, 240, 240, 155)` and 1mm voxel spacing are verified automatically.
3. **Run Real Inference:**
   - Verify that **Demo Mode is OFF** (default setting).
   - Click **"🔬 Run AI Brain Tumor Segmentation"**.
   - Monitor the progress bar during 3D sliding-window convolution (~45–90s on CPU).
4. **Interactive Slice Navigation:**
   - Switch display modes: *Overlay*, *Side-by-Side*, or *Original*.
   - Click **"🎯 Peak Tumor Slice"** to jump directly to axial slice `#82` (maximal lesion cross-section).
   - Toggle sub-region overlays (WT, TC, ET) and inspect the **Ground-Truth Benchmark Evaluation** card showing verified >91% Dice coefficients.
5. **Tumor Analysis:**
   - Navigate to **"📊 Tumor Analysis"** to inspect volumetric measurements ($63.5\text{ cm}^3$), bounding-box dimensions ($78.0 \times 60.0 \times 63.0\text{ mm}$), and sub-region percentage ratios.
6. **Confidence Heatmap:**
   - Navigate to **"🎯 Confidence"** to inspect the voxel-wise probability heatmap and statistical distribution across lesion voxels.
7. **3D Point Cloud:**
   - Navigate to **"🧊 3D Visualization"** to orbit, pan, and zoom the interactive 3D spatial reconstruction.
8. **Clinical PDF Report:**
   - Navigate to **"📄 Clinical Report"**, click **"📄 Build PDF Summary Report"**, then click **"⬇️ Download NeuroVision Clinical Report (PDF)"**.

---

## 🧪 Automated Testing & Verification

NeuroVision includes a dedicated test suite using Python's standard `unittest` framework:

```powershell
# Run fast unit tests (12 tests covering preproc, analysis, viewer, reportLab)
python run_tests.py --fast

# Run all tests including the full MONAI SegResNet benchmark test
python run_tests.py --all
```

All 12 core unit tests execute and pass in ~1.4 seconds. Full testing methodology and verified logs are documented in [docs/TESTING.md](file:///c:/Users/thoma/Downloads/BrainTumorAI/docs/TESTING.md).

---

## 📖 Documentation Index

| Document | Purpose |
| :--- | :--- |
| [docs/SETUP.md](file:///c:/Users/thoma/Downloads/BrainTumorAI/docs/SETUP.md) | Detailed Windows setup, PowerShell execution policy, and dependency configuration |
| [docs/ARCHITECTURE.md](file:///c:/Users/thoma/Downloads/BrainTumorAI/docs/ARCHITECTURE.md) | Technical architecture, MONAI SegResNet design, sliding-window pipeline, and data flow |
| [docs/TESTING.md](file:///c:/Users/thoma/Downloads/BrainTumorAI/docs/TESTING.md) | Test commands, execution outputs, benchmark Dice results, and testing scope boundaries |
| [docs/MENTOR_HANDOVER.md](file:///c:/Users/thoma/Downloads/BrainTumorAI/docs/MENTOR_HANDOVER.md) | Comprehensive mentor handover dossier, file map, known limitations, and roadmap |
| [models/README.md](file:///c:/Users/thoma/Downloads/BrainTumorAI/models/README.md) | Checkpoint specification, model inputs/outputs, SHA256 checksum, and provenance |

---

## ⚠️ Limitations & Responsible-Use Notice

- **Investigational Research Prototype:** NeuroVision is developed strictly for academic, educational, and research evaluation. It is **NOT an FDA-cleared or CE-marked medical device**.
- **No Direct Diagnostic Use:** Segmentation outputs, volumetric measurements, and confidence scores must not be used as primary criteria for surgical resection margins, radiotherapy planning, or clinical treatment decisions without independent certified neuroradiologist review.
- **Preprocessing Assumptions:** The deep learning model expects skull-stripped, co-registered multimodal scans. Unregistered scans, raw unstripped DICOM acquisitions, or severe motion artifacts will degrade boundary accuracy.
- **Scope of Analysis:** NeuroVision does not predict WHO histopathological grade, predict overall survival, or determine molecular/genetic biomarkers (e.g. IDH mutation, 1p/19q codeletion, MGMT promoter methylation).

---

## 🛠️ Troubleshooting & FAQs

### PowerShell Script Restricted Error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Port 8501 Collision
Launch on an alternate port:
```powershell
streamlit run app.py --server.port 8502
```

### CPU Inference Runtime
Sliding-window convolution over $240 \times 240 \times 155$ voxels on CPU typically requires 45–90 seconds. To bypass model execution during rapid UI layout testing, toggle **Demo Mode** ON in the sidebar.

---

## 📜 Provenance & Citations

- **MONAI Framework:** Medical Open Network for AI (Apache 2.0 license) — [https://monai.io](https://monai.io)
- **BraTS Benchmark:**
  - Baid, U., et al. "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification." *Frontiers in Neuroscience*, 15, 2021.
  - Bakas, S., et al. "Advancing The Cancer Genome Atlas glioma analysis with analytics, tissue images, and clinical data." *Scientific Data*, 4, 170117, 2017.
- **Pretrained Checkpoint:** MONAI SegResNet trained on the BraTS challenge cohort, evaluated for research evaluation.
