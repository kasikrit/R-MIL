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
