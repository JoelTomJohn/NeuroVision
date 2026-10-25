# NeuroVision — Windows Setup & Installation Guide

This guide provides step-by-step instructions for installing, configuring, and verifying the **NeuroVision** clinical research prototype on Windows 10 or 11.

---

## 💻 System Prerequisites

| Component | Minimum Requirement | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Windows 10 (64-bit) or Windows 11 | Windows 11 (64-bit) |
| **Python** | Python 3.9.0 to 3.9.13 (Verified on **3.9.5**) | Python 3.9.5 |
| **System Memory (RAM)** | 8 GB RAM | 16 GB RAM |
| **Storage** | 2 GB free disk space | 5 GB SSD storage |
| **Compute Accelerator** | CPU (Multi-core x86_64) | NVIDIA GPU (CUDA 11.8+ / 12.x) with 6+ GB VRAM |

> [!NOTE]
> NeuroVision is thoroughly verified on CPU-only hardware. While sliding-window inference is hardware-accelerated if an NVIDIA GPU with PyTorch CUDA is detected, a discrete GPU is **not required** to run the complete pipeline.

---

## 📥 1. Repository Acquisition

Clone the repository using Git:

```powershell
git clone https://github.com/JoelTomJohn/NeuroVision.git
cd NeuroVision\BrainTumorAI
```

If working within the existing local installation:
```powershell
cd C:\Users\thoma\Downloads\BrainTumorAI\BrainTumorAI
```

---

## 🐍 2. Python Virtual Environment Setup

### A. PowerShell Execution Policy (One-time Windows configuration)
By default, Windows PowerShell restricts script execution. If you encounter an execution policy error when activating the environment, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### B. Create and Activate Virtual Environment
Create a dedicated virtual environment named `venv`:

```powershell
# Create virtual environment with Python 3.9
python -m venv venv

# Activate virtual environment in PowerShell
.\venv\Scripts\Activate.ps1
```

*(For Command Prompt `cmd.exe`, run `venv\Scripts\activate.bat`)*

Verify that the environment points to the isolated Python interpreter:
```powershell
Get-Command python | Select-Object Source
python --version
```
Expected output: `Python 3.9.x`

---

## 📦 3. Dependency Installation

Upgrade `pip` and install the tested package dependencies specified in [requirements.txt](file:///C:/Users/thoma/Downloads/BrainTumorAI/BrainTumorAI/requirements.txt):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Core Dependency Manifest

| Package | Purpose |
| :--- | :--- |
| `torch` (>=2.0.0) | PyTorch deep learning backend for neural network tensor computation |
| `monai` (>=1.3.0) | Medical Open Network for AI (`SegResNet`, `sliding_window_inference`) |
| `nibabel` (>=5.2.0) | Medical neuroimaging I/O for NIfTI (`.nii`, `.nii.gz`) data and header zooms |
| `streamlit` (>=1.32.0) | Interactive reactive web application framework |
| `plotly` (>=5.20.0) | Interactive 2D radiological overlays and 3D point cloud volumetric viewer |
| `reportlab` (>=4.1.0) | Clinical PDF document generation engine |
| `numpy` (>=1.26.0) | High-performance multidimensional array manipulations |
| `scipy` (>=1.10.0) | Spatial morphology and image coordinate processing |
| `pandas` (>=2.2.0) | Structured tabular data representations |
| `Pillow` (>=10.0.0) | Image formatting and thumbnail rendering |

---

## 🔍 4. Verification of Model Checkpoint & Bundled Data

Ensure that the model weights and bundled benchmark scans are in place before launching:

### A. Pretrained Model Checkpoint
Check that `models/model.pt` exists and is ~18 MB:
```powershell
Get-Item .\models\model.pt | Select-Object Name, Length
```
- File: `models/model.pt`
- Expected size: ~18,840,620 bytes (~17.97 MB)

### B. Benchmark BraTS Patient Scan
Check that the bundled benchmark patient (`BraTS-GLI-00000-000`) contains all 5 required sequences:
```powershell
Get-ChildItem .\data\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\BraTS-GLI-00000-000\
```
- `BraTS-GLI-00000-000-t1c.nii.gz` (T1 contrast-enhanced)
- `BraTS-GLI-00000-000-t1n.nii.gz` (T1 native)
- `BraTS-GLI-00000-000-t2w.nii.gz` (T2-weighted)
- `BraTS-GLI-00000-000-t2f.nii.gz` (T2-FLAIR)
- `BraTS-GLI-00000-000-seg.nii.gz` (Expert neuroradiologist ground truth mask)

---

## 🧪 5. Run Verification Tests

Run the included automated test suite to confirm all components are operational:

```powershell
# Run all fast unit tests (12 tests, ~1.5 seconds)
python run_tests.py --fast

# Optional: Run the complete benchmark evaluation including full 3D SegResNet inference (~90 seconds on CPU)
python run_tests.py --all
```

---

## 🚀 6. Launching NeuroVision

Launch the Streamlit web application:

```powershell
streamlit run app.py
```

*Direct execution via Python module:*
```powershell
python -m streamlit run app.py
```

The application will start the local web server and automatically open your default browser at:
```
http://localhost:8501
```

To run on a custom port or in headless mode:
```powershell
streamlit run app.py --server.port 8502 --server.headless true
```

---

## 🛠️ Windows Troubleshooting & FAQs

### Q1: `Activate.ps1 cannot be loaded because running scripts is disabled on this system`
**Cause:** Windows PowerShell script execution policy defaults to `Restricted`.  
**Resolution:** Run PowerShell as Administrator or current user and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q2: Port 8501 is already in use
**Cause:** A previous Streamlit instance or other application is running on port 8501.  
**Resolution:** Launch on an alternate port or terminate the conflicting process:
```powershell
# Identify process using port 8501
Get-NetTCPConnection -LocalPort 8501

# Launch on port 8502 instead
streamlit run app.py --server.port 8502
```

### Q3: PyTorch CPU vs CUDA installation
**Behavior:** By default, NeuroVision utilizes CPU computation (`torch.device("cpu")`). If an NVIDIA GPU is present and CUDA is desired:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
NeuroVision will automatically detect `cuda` and route tensor computations to GPU without any code changes.

### Q4: Long inference times on CPU
**Behavior:** Sliding-window inference on a full $240 \times 240 \times 155$ 4-modality volume on CPU typically takes between 45 and 90 seconds depending on processor speed. This is expected behavior for 3D convolutional architectures with overlapping Gaussian windows. For quick UI navigation during development, **Demo Mode** can be toggled on in the sidebar.
