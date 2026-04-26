# R-MIL: Riemann Geometry-Inspired Multiple Instance Learning for Hematopathology

This repository contains the official implementation of the hybrid AI framework integrating **Riemann Geometry-inspired Multiple Instance Learning (R-MIL)**. The system is designed to provide aleatoric noise protection and clinical abstention (triage) for the automated screening of Iron Deficiency Anemia (IDA) and Thalassemia (THL).

> **Paper Title:** Safety-First Hematological Screening: A Geometry-Constrained AI Pipeline with Clinical Abstention for Differentiating Iron Deficiency Anemia and Thalassemia  
> **Journal:** Pattern Analysis and Applications (Submitted)  
> **Citation:** *Pending*

## Abstract

**Background:** Efficient screening for Iron Deficiency Anemia (IDA) and Thalassemia (THL) is currently hindered by the labor-intensive nature of manual blood smear analysis.

**Objective:** We present a weakly supervised Riemannian Multiple Instance Learning (R-MIL) framework for automated differentiation using a dataset of 159 patients (111 for training and 48 for unseen testing).

**Method:** The architecture employs a two-stage process: manifold curvature energy calibration followed by ensemble-based inference with geometry-constrained aggregation. A critical innovation is the integration of an uncertainty-aware triage mechanism that identifies artifact-compromised cases.

**Results:** The R-MIL approach achieved 0.94 accuracy and 0.96 sensitivity. The system demonstrated exceptional specificity for IDA, providing definitive diagnostic confidence that reduces redundant laboratory testing and patient costs. Statistical validation confirms that the triage logic safely defers artifact-compromised cases to human review instead of forcing low-confidence predictions.

**Conclusion:** By actively managing diagnostic uncertainty, this architecture serves as a highly reliable, confidence-aware screening tool with significant potential to optimize workflows in resource-limited clinical settings.

## Hardware and Software Specifications

* **Python Version:** 3.9.16
* **TensorFlow Version:** 2.11.0 *(Utilized for CNN backbone feature extraction)*
* **PyTorch Version:** 1.13.0 *(Utilized for Q-MIL head, quantum latent projections, and gradient computation)*

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
pip install torch==1.13.0 torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib seaborn yacs tqdm opencv-python Pillow
```

## Available Dataset

The datasets generated and/or analyzed during the current study are not publicly available as they are being utilized for ongoing and future research, but are available from the corresponding author on reasonable request. However, a related dataset with similar attributes is publicly accessible at [https://github.com/kasikrit/IDA-THL-Classification/](https://github.com/kasikrit/IDA-THL-Classification/).

## R-MIL Pipeline Execution

### 1. Configure Pipeline Parameters
Modify the architectural parameters, learning rates, and directory paths via the central `config.py` file utilizing `yacs`.

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
