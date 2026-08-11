# FSTS-Net

Pytorch implementation of **FSTS-Net** for semantic change detection (SCD) and binary change detection (BCD) in multi-temporal remote sensing images.

## Environment Setup

### 1. Create and activate conda environment (Python 3.10, CUDA 11.8)

```bash
conda create -n FSTS-Net python=3.10
conda activate FSTS-Net
```

### 2. Install PyTorch 2.1.1 with CUDA 11.8

```bash
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. Install base dependencies

```bash
pip install -r requirements.txt
```

### 4. Install additional required packages

```bash
pip install mamba-ssm==1.0.1
pip install causal-conv1d==1.2.1
pip install selective_scan==0.0.2
pip install PyWavelets
pip install thop
```

### 5. Install TreeScan (C++ CUDA extension)

```bash
cd GrootV/third-party/TreeScan
pip install -v -e .
```

> **Note:** CUDA 11.8 is recommended. Make sure `CUDA_HOME` is properly set and a compatible GCC/G++ compiler is available.

## Project Structure

```
FSTS-Net/
├── datasets/                    # Data loading modules
│   ├── MultiSiamese_RS_ST_TL_BRIGHT.py      # BRIGHT dataset loader
│   ├── MultiSiamese_RS_ST_TL_BRIGHT_BCD.py  # Wuhan BCD dataset loader
│   ├── make_data_loader.py                   # NewDataset/SN6 loader
│   └── imutils.py
├── models/                      # Model architectures
│   ├── Proposed_method.py       # Main model (FSTS-Net)
│   ├── Multimodal_SCD_0609.py   # Model variant
│   ├── vmamba.py                # VSSM backbone (state space model)
│   ├── dual_vmamba.py           # Dual-branch VSSM for SCD
│   ├── Backbones/               # CNN backbones (ResNet variants)
│   ├── Decoders/                # Segmentation & change detection decoders
│   ├── Modules/                 # CIEM (Cross-modal Interaction & Enhancement)
│   ├── WTFMBlock0609.py         # Wavelet-based feature modulation
│   └── sigma/                   # Sigma utilities
├── GrootV/                      # GrootV graph optimization module
│   └── classification/models/
│       ├── grootv.py / grootv0609.py         # GrootV layers
│       ├── tree_scanning.py / tree_scanning0609.py
│       └── tree_scan_utils/                  # Tree scan CUDA kernels
├── utils/                       # Utility functions
│   ├── metrics.py               # Evaluation metrics (IoU, F1, etc.)
│   ├── metric.py                # BCD-specific metrics
│   ├── loss.py / loss_functions.py / lovasz_loss.py  # Loss functions
│   ├── palette.py / colormap.py # Visualization
│   └── misc.py / misc0609.py    # Misc helpers
├── train_BRIGHT.py              # Training on BRIGHT dataset (SCD, 4-class)
├── train_NewDataset.py          # Training on NewDataset/SN6 (SCD, 7-class)
├── train_wuhan.py               # Training on Wuhan dataset (BCD, 2-class)
├── inference_BRIGHT.py          # Inference on BRIGHT dataset
├── inference_SN6.py             # Inference on NewDataset/SN6
└── inference_wuhan.py           # Inference on Wuhan dataset
```

## Supported Datasets

| Dataset    | Task | Classes | Training Script    | Inference Script       |
|------------|------|---------|--------------------|------------------------|
| BRIGHT     | SCD  | 4       | train_BRIGHT.py    | inference_BRIGHT.py    |
| Wuhan-Het  | BCD  | 2       | train_wuhan.py     | inference_wuhan.py     |
| Delt-SN6   | SCD  | 7       | train_NewDataset.py | inference_SN6.py       |

## Pretrained Weights

The training weights for each dataset can be downloaded via **Baidu Cloud Disk (百度云盘)**. Please refer to the download link and extraction code provided with the project release.

## Usage

### Training

Before running, modify the data paths inside each training script to point to your local dataset directories.

```bash
# BRIGHT dataset (4-class semantic change detection)
python train_BRIGHT.py

# Wuhan-Het dataset (binary change detection)
python train_wuhan.py

# Delt-SN6 dataset (7-class semantic change detection)
python train_NewDataset.py
```

### Inference

Update `--model_path`, `--test_dataset_path`, and `--test_data_list_path` for your setup.

```bash
# BRIGHT dataset inference
python inference_BRIGHT.py --model_path /path/to/model.pth

# Wuhan-Het dataset inference
python inference_wuhan.py --load_from /path/to/model.pth

# Delt-SN6 dataset inference
python inference_SN6.py --model_path /path/to/model.pth
```

### Data Format

Datasets are organized with paired pre-change / post-change images and corresponding labels. Refer to the dataset loader files in `datasets/` for the specific format expected for each dataset.
