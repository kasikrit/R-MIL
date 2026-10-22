# A Geometry-Constrained Multiple-Instance Learning Pipeline with Clinical Abstention for Differentiating Iron Deficiency Anemia and Thalassemia on Peripheral Blood Smears

This repository contains the official implementation of the hybrid AI framework integrating **Riemann Geometry-inspired Multiple Instance Learning (R-MIL)**. The system is designed to provide aleatoric noise protection and clinical abstention (triage) for the automated screening of Iron Deficiency Anemia (IDA) and Thalassemia (THL).

<!-- > **Paper Title:** Geometry-Constrained AI Pipeline with Clinical Abstention for Differentiating Anemia Subtypes  -->
> **Journal:** Applied Intelligence (Accepted)  
> **Citation:** *pending*

## Abstract

Manual blood smear analysis for differentiating Iron Deficiency Anemia (IDA) and Thalassemia (THL) is labor-intensive. We present a weakly supervised Riemannian Multiple Instance Learning (R-MIL) framework for automated differentiation using a dataset of 159 patients (111 training, 48 unseen testing). The architecture employs a two-stage process combining manifold curvature energy calibration with ensemble-based inference and geometry-constrained aggregation. To manage weak-label noise and morphological artifacts, we introduce a curvature-aware triage mechanism that isolates highly confident predictions while deferring anomalous out-of-distribution (OOD) or uncertain cases to expert review. On the independent test cohort, the triage system achieved an autonomous resolution rate (coverage) of 64.6\%. Within this high-confidence subset, the framework yielded an accuracy of 0.87, a sensitivity of 1.00 for THL, and a Negative Predictive Value (NPV) of 1.00. Statistical validation confirmed that the abstention logic successfully isolated ambiguous morphology without disrupting the predictive capability of straightforward cases. By actively managing uncertainty, this architecture serves as a highly reliable human-in-the-loop decision support tool with the potential to optimize clinical screening workflows in resource-limited settings.

## Hardware and Software Specifications

* **Python Version:** 3.9.16
* **TensorFlow Version:** 2.11.0 *(Utilized for CNN backbone feature extraction)*
<!-- * **PyTorch Version:** 1.13.0 *(Utilized for Q-MIL head, quantum latent projections, and gradient computation)* -->

## Creating a Python Environment

We recommend using Anaconda or Miniconda to manage the environment.

```bash
conda create -n rmil_env python=3.9.16
conda activate rmil_env
```

## Installing Required Software Packages

Ensure to install the correct PyTorch wheels for your specific CUDA version (e.g., CUDA 11.7 for PyTorch 1.13).

```bash
pip install tensorflow==2.11.0
pip install pandas numpy scikit-learn matplotlib seaborn yacs tqdm opencv-python Pillow
```

## Data

### Cell images (public)

The segmented red-blood-cell images are published on figshare:

- **DOI:** [10.6084/m9.figshare.32788653](https://doi.org/10.6084/m9.figshare.32788653)
- **Files:** six zip parts, each under 3 GB, plus `MD5SUMS.txt` and `README_parts.txt`. Download all six and extract them into **one folder** to rebuild the dataset:
  ```bash
  for f in IDA-THL-Segmented-Cells-02-2026_part*.zip; do unzip -q "$f"; done
  ```
- **Contents:** 291 patient folders (151 iron deficiency anemia, 140 thalassemia) from 281 patients, with 261,087 cell images.

| Part | File | Patients | Size | MD5 |
|---|---|---|---|---|
| 1 | `IDA-THL-Segmented-Cells-02-2026_part1-of-6_IDA-001-050.zip` | IDA-001–050 | 2.39 GB | `623acc17387c3dd91f046fef123c9849` |
| 2 | `IDA-THL-Segmented-Cells-02-2026_part2-of-6_IDA-051-101.zip` | IDA-051–101 | 2.36 GB | `d293b616cebf3e4849cc85441c3f5224` |
| 3 | `IDA-THL-Segmented-Cells-02-2026_part3-of-6_IDA-102-151.zip` | IDA-102–151 | 2.38 GB | `61ea948e880c28548524755100cb051a` |
| 4 | `IDA-THL-Segmented-Cells-02-2026_part4-of-6_THL-001-047.zip` | THL-001–047 | 2.73 GB | `2c9d0d3048eb57aa7434f6e190acf4b4` |
| 5 | `IDA-THL-Segmented-Cells-02-2026_part5-of-6_THL-048-090.zip` | THL-048–090 | 2.69 GB | `bf8ca7011a4bb999ad30129546f05577` |
| 6 | `IDA-THL-Segmented-Cells-02-2026_part6-of-6_THL-091-140.zip` | THL-091–140 | 2.72 GB | `d0fbc0da6bcedfac896a9647f69b30c1` |

Together the parts hold the same 261,087 files as the single archive `IDA-THL-Segmented-Cells-02-2026.zip` (MD5 `d2f896c59984890baaeb5e21253061e0`), verified file by file.

The release follows a Class → Patient → Slide → Cell schema:

```
IDA-THL-Segmented-Cells-02-2026/
├── 0/                      # IDA
│   └── IDA-001/cells/DSC10132/IDA-001-DSC10132-001.png
└── 1/                      # THL
    └── THL-001/cells/...
```

The patient IDs in the paper's figures (e.g. IDA-063, THL-097) are the folder names of this release.

Nine patients were imaged in more than one acquisition batch and so have more than one folder. The figshare description lists them. For strictly patient-level splits, group those folders together.

**Folder layout used by the code.** The experiments were run on the original, non-anonymized tree, which is organized as `<batch>/<class>/<patient>/normalized/<slide>-cells-256/*.png`. The figshare release was generated from that tree by `tcg_1.py` (`RUN.PUBLISH_DATASET`).

To run the code on the figshare release, adapt the scanner in `Share/mygears.py` (the `glob_pattern` and the path-part indices) to the `<class>/<patient>/cells/<slide>/*.png` layout. The patient ID is then the folder name.

## R-MIL Pipeline Execution

### 0. Trained Baseline Models of Dataset 2025 as Feature Extractors
Download the trained baseline models (ConvNeXtLarge) at [Figshare](https://doi.org/10.6084/m9.figshare.32101114) and locate at C.BASEMODEL_PATH

To build your own feature extractors using a custom dataset, execute the baseline training pipeline `basemodel-train-20251130.py` using the settings defined in `configsAneTunedHP_2.py`.

### 1. Configure Pipeline Parameters
Modify the architectural parameters, learning rates, and directory paths via the central `configsCell_mil_stage2.py` file utilizing `yacs`.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
from datetime import datetime 
from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.BASE = str(platform.system())

if platform.system() == 'Windows':
    _C.BASEPATH = r'.'
    _C.DATASET = "D:\\THL-g" 
    _C.BASEMODEL_PATH = os.path.join(r"D:\THL-g", 'models')
    _C.MODEL_PATH = os.path.join(r"D:\THL-g", 'models')
    _C.verbose = 1

elif platform.system() == 'Darwin': 
    _C.BASEPATH = '/Users/kasikritdamkliang/Datasets/THL-g' 
    _C.DATASET = '/Users/kasikritdamkliang/Datasets/THL-g/Seg-cells'
    _C.BASEMODEL_PATH = os.path.join(_C.BASEPATH, 'models')
    _C.MODEL_PATH = os.path.join(_C.BASEPATH, 'models-qmil')  
    _C.verbose = 1

elif platform.system() == 'Linux':
    _C.BASEPATH = '/home/11001214/kasikritD/THL-g' 
    _C.DATASET = '/home/11001214/kasikritD/THL-g/Seg-cells'
    _C.BASEMODEL_PATH = os.path.join(_C.BASEPATH, 'models')
    _C.MODEL_PATH = os.path.join(_C.BASEPATH, 'models')
    _C.verbose = 2
    
_C.base_dir_list = [
    '20240920', '20240930', '20241001', '20241020', '20241121',
    '20241210', '20241228', '20250111', '20250331', '20250727', '20250914'
]

_C.SAVEPATH = os.path.join(_C.BASEPATH, 'models')
_C.DATASET_YEAR='DATASET2025'
_C.DATA.SEED = 1337
_C.DATA.C = 3
_C.DATA.W = _C.DATA.H = 256
_C.DATA.IMAGE_SIZE = _C.DATA.W
_C.DATA.DIMENSION = (_C.DATA.W, _C.DATA.H, _C.DATA.C)
_C.DATA.TEST_SIZE = 0.30
_C.DATA.CLASSES = [0, 1]

_C.DATA.CLASS_LABELS = ['IDA', 'THL']
_C.DATA.CLASSDICT = [(0, "IDA"), (1, "THL")]
_C.DATA.COLOR = 'RGB'

_C.MODEL = CN()
_C.MODEL.NAMES = ['ConvNeXtLarge']   
_C.MODEL.NUM_CLASSES = 2

_C.TRAIN = CN()

if platform.system() in ['Windows', 'Darwin']:
    _C.TRAIN.verbose = 1
elif platform.system() == 'Linux':    
    _C.TRAIN.verbose = 2
        
_C.TRAIN.DATETIME = datetime.now().strftime("%Y%m%d-%H%M")
_C.TRAIN.TL = False
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.EPOCHS = 150
_C.TRAIN.ACTIVATION = 'relu'  
_C.TRAIN.LR = 1.3958e-05 
_C.TRAIN.DROPOUT = 0.4

_C.TRAIN.LR_FACTOR = 0.5  
_C.TRAIN.Reduce_LR_patience = 2
_C.TRAIN.stop_patience = 20 
_C.TRAIN.l2_reg = 3.1976e-05
_C.DATA.val_ratio = 0.30

stage2_lr = 3e-07
stage3_lr = 3e-05
head_lr  =  3e-05

_C.TRAIN.base_lr = [stage2_lr, stage3_lr, head_lr]
_C.TRAIN.lr_mode = "cosine"
_C.TRAIN.Class_weight = True

if "ViT" in _C.MODEL.NAMES:
    _C.TRAIN.ACTIVATION = 'gelu'

_C.DATA.VERIFY = True
_C.DATA.N_SPLIT = 5
_C.TRAIN.FOLDS = [0, 1, 2, 3, 4]
_C.TRAIN.SAMPLE = False

if _C.TRAIN.SAMPLE and platform.system() == 'Darwin':
    _C.TRAIN.SAMPLE_SIZE = 32
    _C.TRAIN.SAMPLE_SIZE_VAL = int(_C.TRAIN.SAMPLE_SIZE * _C.DATA.val_ratio)
    _C.TRAIN.SAMPLE_SIZE_TEST = 4
    _C.TRAIN.BATCH_SIZE = 4
    _C.TRAIN.EPOCHS = 4
    _C.DATA.N_SPLIT = 3
    _C.TRAIN.Actual_Fold = 2
    _C.TRAIN.verbose = 1

elif _C.TRAIN.SAMPLE and platform.system() == 'Linux':
    _C.TRAIN.SAMPLE_SIZE = 2048
    _C.TRAIN.SAMPLE_SIZE_VAL = int(_C.TRAIN.SAMPLE_SIZE * _C.DATA.val_ratio)
    _C.TRAIN.SAMPLE_SIZE_TEST = 600
    _C.TRAIN.BATCH_SIZE = 128
    _C.TRAIN.EPOCHS = 20
    _C.DATA.N_SPLIT = 3
    _C.TRAIN.Actual_Fold = 2
    _C.TRAIN.verbose = 1

_C.TRAIN.Enable = True
_C.PILOT = not _C.TRAIN.Enable

_C.TRAIN.Evaluate_Val = True
_C.TRAIN.Evaluate_Test = True

_C.SAVE = True
_C.SAVEHIST = _C.TRAIN.Enable
    
def get_config():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()

```

### 2. Start Ensemble Model Training
Execute the training phase to extract classical CNN features, project them into the Poincaré ball (Riemannian manifold), and train the independent R-MIL ensemble folds using curvature-aware attention.

```bash
python main.py --mode train --data_dir ./data/ --weights_dir ./models/
```

## Patient-Level Classification: Manifold-Based Triage

### 1. Run Threshold Calibration (Phase 1B)
Executes Out-of-Fold (OOF) evaluation to establish the strict curvature energy boundary ($$E_{limit}$$) on unseen validation patients, entirely preventing test-set data leakage while filtering aleatoric noise.

### 2. Independent Test Inference (Phase 3)
Evaluates the soft-voting ensemble on the held-out test set and enforces the double-verification clinical triage gate based on the calibrated $$H_{limit}$$.

```bash
python main.py --mode test
```

### 3. Generate Explainable AI Visualizations (Phase 4)
Produces the 3-panel XAI suite for a targeted patient, directly rendering the mathematical etiology of the manifold topology.

* **Panel 1:** Morphological Latent Space (2D Isomap Projection)
* **Panel 2:** Riemannian Curvature Energy Landscape 
* **Panel 3:** Saliency Rescue Gate Activation (High-Energy Diagnostic Patches)

```bash
python main.py --mode xai
```

## Output
Execution of the phases above will automatically compile and save the evaluation metrics, triage performance, manifold parameters (`.pkl`), and `patient_predictions_mil_val.csv` in the designated `./models/` directory.
```
