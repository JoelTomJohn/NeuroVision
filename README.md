# 🧠 NeuroVision — AI-Assisted Brain MRI Analysis

**NeuroVision** is a clinical research prototype for multimodal 3D brain MRI segmentation, quantitative volumetric analysis, confidence quantification, and standardized reporting. Powered by a deep learning **MONAI SegResNet** architecture pretrained on the BraTS (Brain Tumor Segmentation) benchmark, NeuroVision automatically delineates whole tumor, tumor core, and enhancing tumor sub-compartments.

---

## 🌟 Key Features

1. **Multimodal 3D MRI Segmentation:**
   - Simultaneous processing of four co-registered diagnostic MRI sequences: **T1c** (contrast-enhanced), **T1** (native), **T2** (T2-weighted), and **FLAIR** (fluid-attenuated inversion recovery).
   - Delineation of standard BraTS clinical sub-regions:
     - **Whole Tumor (WT):** Peritumoral vasogenic edema + Tumor core.
     - **Tumor Core (TC):** Necrotic / cystic non-enhancing core + Active enhancing tumor.
     - **Enhancing Tumor (ET):** Active contrast-enhancing hypervascular rim and nodular core.

2. **Interactive Radiological Viewer:**
   - Multi-mode axial visualization: *Original MRI*, *AI Segmentation*, *Radiological Overlay*, and *Side-by-Side Comparison*.
   - Multi-region overlay toggling with distinct clinical color swatches.
   - Synchronized slice navigation slider, previous/next controls, and an instant **"🎯 Peak Tumor Slice"** jump button.

3. **Quantitative Volumetric & Morphometric Analysis:**
   - Accurate calculation of tumor volume in **cubic centimeters ($\text{cm}^3$)** using exact NIfTI voxel spacing header metadata ($s_x \times s_y \times s_z$).
   - Physical bounding-box orthogonal dimensions ($L \times W \times H$ in $\text{mm}$) along anatomical planes.
   - Sub-compartment volume fractions (Enhancing Tumor % and Tumor Core % relative to Whole Tumor).

4. **Honest Confidence & Explainability:**
   - Spatial voxel-wise probability heatmap rendered via Plotly.
   - Summary statistics (Mean, Min, Max, and Standard Deviation) across predicted tumor voxels.
   - Clear disclosure distinguishing heuristic probability activations from Bayesian calibrated posterior uncertainty.

5. **Interactive 3D Volumetric Exploration:**
   - Hardware-accelerated 3D point-cloud exploration.
   - Multi-layer spatial separation with orbital rotation, panning, and zoom controls.

6. **Standardized Clinical PDF Reporting:**
   - One-click compilation of downloadable clinical summary reports using ReportLab.
   - Complete scan metadata, volumetric findings, confidence metrics, embedded key axial slice thumbnail, and regulatory disclaimers.

7. **One-Click Benchmark Demonstration:**
   - Pre-bundled benchmark patient scan (`BraTS-GLI-00000-000`) for instantaneous presentation and testing.
   - Ground-truth evaluation module calculating real Dice Similarity Coefficients against expert neuroradiologist annotations.

---

## 🏗️ Model Architecture & Verification

- **Architecture:** MONAI SegResNet (3D residual encoder-decoder)
- **Input Channels:** 4 (Channel 0: T1c, Channel 1: T1, Channel 2: T2, Channel 3: FLAIR)
- **Output Channels:** 3 (Sigmoid activations for Tumor Core, Whole Tumor, Enhancing Tumor)
- **Initial Filters:** 16
- **Encoder Blocks:** (1, 2, 2, 4) with 0.2 dropout
- **Decoder Blocks:** (1, 1, 1)
- **Pretrained Weights:** `models/model.pt` (18 MB)
- **Verified Benchmark Scores (BraTS-GLI-00000-000):**
  - **Whole Tumor (WT) Dice:** **0.9184** (91.8%)
  - **Tumor Core (TC) Dice:** **0.9748** (97.5%)
  - **Enhancing Tumor (ET) Dice:** **0.9356** (93.6%)
- **Hardware Optimization:** Verified on CPU with 8 GB RAM and Intel integrated graphics (inference runtime: ~45 seconds). Discrete NVIDIA GPU is supported via PyTorch CUDA when available, but not required.

---

## 📋 Required MRI Modalities & File Formats

NeuroVision requires four co-registered, skull-stripped 3D MRI volumes in NIfTI format (`.nii` or `.nii.gz`):

| Sequence | Clinical Role | Signal Characteristic |
| :--- | :--- | :--- |
| **T1c** | Delineates active breakdown of blood-brain barrier | Hyperintense contrast enhancement |
| **T1** | Anatomical reference sequence | Gray/white matter parenchymal boundary |
| **T2** | Highlights abnormal tissue free-water content | Hyperintense edema & fluid collection |
| **FLAIR** | Suppresses normal cerebrospinal fluid (CSF) signal | Hyperintense peritumoral vasogenic edema |

*Note: Ground-truth segmentation masks (e.g. `*-seg.nii.gz`) must NEVER be uploaded as one of the four MRI input channels.*

---

## 🚀 Quickstart & Windows Launch Instructions

### 1. Prerequisites
- Windows 10 / 11
- Python 3.9.5 (or Python 3.9+)
- 8 GB RAM minimum

### 2. Activate Virtual Environment & Launch
Open PowerShell in the project directory:

```powershell
# Navigate to the project root
cd c:\Users\thoma\Downloads\BrainTumorAI\BrainTumorAI

# Activate existing Python 3.9 virtual environment
.\venv\Scripts\Activate.ps1

# Launch the NeuroVision Streamlit application
streamlit run app.py
```

*Alternatively, launch directly without activating the shell:*
```powershell
& c:\Users\thoma\Downloads\BrainTumorAI\BrainTumorAI\venv\Scripts\python.exe -m streamlit run c:\Users\thoma\Downloads\BrainTumorAI\BrainTumorAI\app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🖥️ Presentation Demonstration Checklist

For a smooth, professional demonstration on October 27:

1. **Dashboard Overview:**
   - Launch app; point out the **NeuroVision** branding, tagline, and clinical aesthetic.
   - Show the 4 required MRI modality cards (T1c, T1, T2, FLAIR) and the feature cards.
   - Click **"🚀 Start New Analysis"** to navigate to the MRI Analysis page.

2. **Load Benchmark Sample:**
   - On the MRI Analysis page, click **"⚡ Quick-Load BraTS Benchmark Scan"**.
   - Point out that all 4 modalities (`BraTS-GLI-00000-000`) load instantly with verified dimensions `(4, 240, 240, 155)`.

3. **Run Real Model Inference:**
   - Note that **Demo Mode is OFF by default**, ensuring genuine deep learning inference.
   - Click **"🔬 Run AI Brain Tumor Segmentation"**.
   - Show the progress spinner (~45s on CPU).
   - Once completed, point out the **MONAI SegResNet — Verified Active Inference** badge.

4. **Explore Interactive Slice Viewer:**
   - Switch display mode to **Overlay** or **Side-by-side**.
   - Click **"🎯 Peak Tumor"** button to jump directly to axial slice `#82`, the cross-section with maximum lesion area.
   - Toggle between **Composite (All Regions)**, **Whole Tumor (WT)**, **Tumor Core (TC)**, and **Enhancing Tumor (ET)** with the interactive color legend.
   - Expand the **Ground-Truth Benchmark Evaluation** card to demonstrate the verified Dice metrics (**WT: 91.8%**, **TC: 97.5%**, **ET: 93.6%**).

5. **Review Quantitative Tumor Analysis:**
   - Navigate to **"📊 Tumor Analysis"**.
   - Show Whole Tumor volume (e.g. `63.48 cm³`), bounding-box dimensions ($78.0 \times 60.0 \times 63.0\text{ mm}$), and sub-region ratios (ET is 48.0% of WT).

6. **Demonstrate Confidence & Explainability:**
   - Navigate to **"🎯 Confidence"**.
   - Inspect the Viridis spatial probability heatmap and slider.
   - Review the honest explainability panel distinguishing spatial heuristic probabilities from clinical diagnostic certainty.

7. **Inspect 3D Volumetric Point Cloud:**
   - Navigate to **"🧊 3D Visualization"**.
   - Click and orbit the 3D multi-region point cloud, showing Enhancing Tumor in Crimson Red, Tumor Core in Orange, and Edema in Teal.

8. **Generate & Download PDF Report:**
   - Navigate to **"📄 Clinical Report"**.
   - Click **"📄 Build PDF Summary Report"**.
   - Click **"⬇️ Download NeuroVision Clinical Report (PDF)"** and open the resulting PDF to show the clean clinical layout, tables, embedded axial snapshot, and research disclaimer.

---

## 🧪 Tests Performed & Verification Results

All tests have been run and verified on Python 3.9.5 (Windows):

| Test Suite | Target Component | Result | Notes |
| :--- | :--- | :---: | :--- |
| **Syntax & Imports** | `app.py`, `backend/`, `frontend/` | **PASSED** | Clean execution with no import cycle or missing modules |
| **Model Architecture** | `backend/inference.py` | **PASSED** | Validated SegResNet parameter matching with `model.pt` |
| **CPU Sliding-Window** | `monai.inferers.sliding_window` | **PASSED** | Evaluated on shape `(1, 4, 240, 240, 155)` without OOM |
| **Real Inference Benchmark** | `BraTS-GLI-00000-000` | **PASSED** | WT Dice: 0.9184, TC Dice: 0.9748, ET Dice: 0.9356 |
| **Modality Stacking & Zooms** | `backend/preprocessing.py` | **PASSED** | Verified T1c, T1, T2, FLAIR order and 1mm spacing |
| **Multi-Region Metrics** | `backend/analysis.py` | **PASSED** | Correct volume ($\text{cm}^3$) and bounding box extents ($\text{mm}$) |
| **Plotly 2D/3D Rendering** | `frontend/viewer.py` | **PASSED** | Composite overlays, legends, and multi-region 3D clouds |
| **ReportLab PDF Generation** | `backend/report.py` | **PASSED** | Clean multi-region tables, metadata, and embedded slice image |
| **Demo Mode Toggle** | `app.py` & sidebar | **PASSED** | Honest labeling, no silent fallbacks on real failures |

---

## ⚠️ Limitations & Responsible-Use Notice

- **Investigational Research Prototype:** NeuroVision is developed strictly for academic, educational, and research evaluation. It is **NOT an FDA-cleared or CE-marked medical device**.
- **No Direct Diagnostic Use:** Segmentation outputs, volumetric measurements, and confidence scores must not be used as primary criteria for surgical planning, radiotherapy margins, or patient treatment decisions without certified radiologist review.
- **Preprocessing Assumptions:** The deep learning model expects skull-stripped, co-registered multimodal scans. Unregistered scans, severe patient motion, or raw unstripped DICOM acquisitions may degrade boundary accuracy.
- **Scope of Analysis:** NeuroVision does not predict WHO tumor grade, predict survival, or determine molecular/genetic biomarkers (e.g. IDH mutation, MGMT promoter methylation).

---

## 🔮 Future Work & Roadmap

1. **nnU-Net v2 Framework Integration:** Implement automated self-configuring preprocessing, data augmentation pipelines, and dynamic network topology planning based on the nnU-Net v2 framework.
2. **DICOM Ingestion Pipeline:** Direct import and automated conversion of multi-series DICOM studies via `pydicom` / `SimpleITK`.
3. **Calibrated Uncertainty Ensembles:** Implementation of Monte Carlo Dropout or deep ensembles to provide rigorous epistemic uncertainty boundaries.
4. **Longitudinal Response Tracking:** Automatic co-registration and subtraction across pre- and post-operative scans to track response under RANO criteria.
