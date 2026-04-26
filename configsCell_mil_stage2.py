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
    _C.BASEPATH = r"H:\My Drive\THL-g"
    _C.DATASET = r"H:\My Drive\THL-g"
    _C.SAVE_PATH = os.path.join(_C.BASEPATH, 'Riemannian-Geometry')
    _C.BASEMODEL_PATH = os.path.join(r"D:\THL-g", 'models')
    _C.MODEL_PATH = os.path.join(r"D:\THL-g", 'models', 'mil_model_3_stage2')
    _C.MIL_MODEL = 'mil-model-3'
    _C.verbose = 1

elif platform.system() == 'Darwin':
    _C.BASEPATH = '.' 
    _C.DATASET = '/Users/kasikritdamkliang/Datasets/THL-g/Seg-cells' 
    _C.BASEMODEL_PATH = '/Users/kasikritdamkliang/Datasets/THL-g/models/'  
    _C.MODEL_PATH = '/Users/kasikritdamkliang/Datasets/THL-g/models-rmil'  
    _C.MIL_MODEL = 'mil-model-3'
    _C.SAVE_PATH = '.' 
    _C.verbose = 1
    
elif platform.system() == 'Linux':
    _C.BASEPATH = '/home/11001214/kasikritD/THL-g' 
    _C.DATASET = '/home/11001214/kasikritD/THL-g/Seg-cells' 
    _C.SAVE_PATH = os.path.join(_C.BASEPATH, 'Riemannian-Geometry')
    _C.BASEMODEL_PATH = os.path.join(_C.BASEPATH, 'models')
    _C.MODEL_PATH = os.path.join(_C.SAVE_PATH, 'models', 'mil_model_3_stage2')
    _C.MIL_MODEL = 'mil-model-3'
    _C.verbose = 2

_C.base_dir_list = [
    '20240920', '20240930', '20241001', '20241020', '20241121', 
    '20241210', '20241228', '20250111', '20250331', '20250727', '20250914'
]

_C.DATA.SEED = 42
_C.DATA.C = 3
_C.DATA.W = _C.DATA.H = 256
_C.DATA.IMAGE_SIZE = _C.DATA.W
_C.DATA.DIMENSION = (_C.DATA.W, _C.DATA.H, _C.DATA.C)
_C.DATA.TEST_SIZE = 0.3
_C.DATA.CLASS_LABELS = ['IDA', 'THL']
_C.DATA.CLASSES = [0, 1]
_C.DATA.CLASSDICT = [(0, "IDA"), (1, "THL")]
_C.DATA.val_ratio = 0.3
_C.DATA.COLOR = 'BGR' 
_C.DATA.VERIFY = False

_C.MODEL = CN()
_C.MODEL.SAVEPATH = os.path.join(_C.BASEPATH, 'models')
_C.MODEL.NAMES = ['ConvNeXtLarge']   
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.FEATURE_DIM = 16

_C.TRAIN = CN()
_C.TRAIN.TL = False
_C.TRAIN.DATETIME = datetime.now().strftime("%Y%m%d-%H%M")
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.LR = 5e-5
_C.TRAIN.WEIGHT_DECAY = 1e-05
_C.TRAIN.CLIP_NORM = 1.0
_C.TRAIN.LR_FACTOR = 0.5
_C.TRAIN.EPOCHS = 60
_C.TRAIN.DROPOUT = 0.1
_C.TRAIN.reduceLR_patience = 8   
_C.TRAIN.stop_patience = 20      
_C.DATA.N_SPLIT = 5
_C.TRAIN.Actual_Fold = 4 
_C.TRAIN.Feature_Layer_Index = 296 
_C.TRAIN.SAMPLE = False
_C.TRAIN.ACTIVATION = 'gelu'

if platform.system() in ['Darwin', 'Windows']:    
    _C.TRAIN.verbose = 1
elif platform.system() == 'Linux':    
    _C.TRAIN.verbose = 2

if _C.TRAIN.SAMPLE and platform.system() == 'Windows':
    _C.TRAIN.SAMPLE_SIZE = 16
    _C.TRAIN.SAMPLE_SIZE_VAL = int(_C.TRAIN.SAMPLE_SIZE * _C.DATA.val_ratio)
    _C.TRAIN.SAMPLE_SIZE_TEST = 8 
    _C.TRAIN.BATCH_SIZE = 2
    _C.TRAIN.EPOCHS = 2
    _C.DATA.N_SPLIT = 3
    _C.TRAIN.Actual_Fold = 0
    _C.TRAIN.verbose = 1
    
elif _C.TRAIN.SAMPLE and platform.system() == 'Linux':
    _C.TRAIN.SAMPLE_SIZE = 1024
    _C.TRAIN.SAMPLE_SIZE_VAL = int(_C.TRAIN.SAMPLE_SIZE * _C.DATA.val_ratio)
    _C.TRAIN.SAMPLE_SIZE_TEST = 256
    _C.TRAIN.BATCH_SIZE = 32
    _C.TRAIN.EPOCHS = 60
    _C.DATA.N_SPLIT = 3
    _C.TRAIN.Actual_Fold = 2
    _C.TRAIN.verbose = 1

_C.TRAIN.Enable = True 
_C.PILOT = not _C.TRAIN.Enable
_C.TRAIN.Phase3 = True

_C.Evaluate_Val = False
_C.find_ood = False
_C.find_threshold = False  
_C.PLOT_CURVATURE = False
_C.Evaluate_Test = False
_C.n_select = 3
_C.External_Infer = False
_C.Performance_Report = False
_C.Inference = False
_C.Inference_Audit = False
_C.Inference_Report = False
_C.External_Patch_Audit = False
_C.Performance_Comparison = False
_C.SAVE = True
    
def get_config():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()
    