#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:42:08 2024

@author: kasikritdamkliang
"""
import os, platform
from datetime import datetime 
from yacs.config import CfgNode as CN
from pathlib import Path
# Define default configuration
_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

_C.BASE = str(platform.system())
if platform.system() == 'Windows':  
    _C.BASEPATH = r"H:\My Drive\THL-g"
    _C.DATASET = r"H:\My Drive\THL-g"
    # _C.DATASET = r"D:\AneRBC_dataset\AneRBC-II"
    _C.SAVE_PATH = os.path.join(_C.BASEPATH, 'Riemannian-Geometry')
    _C.BASEMODEL_PATH = os.path.join(r"D:\THL-g", 'models')
    _C.MODEL_PATH = os.path.join("D:\THL-g", 'models', 'mil_model_1')
    # _C.MODEL_PATH = os.path.join("D:\THL-g", 'models', 'mil_model_3')
    _C.MIL_MODEL = 'mil-model-3'
    # _C.DATASET = r"H:\My Drive\THL"
    _C.verbose = 1

if platform.system() == 'Darwin':
    _C.BASEPATH = '.'  
    _C.DATASET = '/Users/kasikritdamkliang/Datasets/Seg-Cells'
    _C.BASEMODEL_PATH = '/Users/kasikritdamkliang/Datasets/models/'  
    # _C.MODEL_PATH = '/Users/kasikritdamkliang/Datasets/models-eucli'  
    _C.MODEL_PATH = '/Users/kasikritdamkliang/Datasets/models-rmil'
    _C.MIL_MODEL = 'eucli-mil-model'
    _C.SAVE_PATH = '.' 
    _C.verbose = 1
    
if platform.system() == 'Linux':
    _C.BASEPATH = '/home/11001214/kasikritD/THL-g' 
    _C.DATASET = '/home/11001214/kasikritD/THL-g/Seg-Cells' 
    # _C.DATASET = "/home/11001214/kasikritD/THL/AneRBC_dataset/AneRBC-II"
    _C.SAVE_PATH = os.path.join(_C.BASEPATH, 'R-MIL')
    _C.BASEMODEL_PATH = os.path.join(_C.SAVE_PATH, 'models')
    _C.MODEL_PATH = os.path.join(_C.SAVE_PATH, 'models', 'rmil_model_3')
    _C.ANE_PATH = '/home/11001214/kasikritD/THL'
    _C.MIL_MODEL = 'rmil-model-3'
    _C.verbose = 2
# _C.DATA.KAPPA_PATH = _C.DATA.FEATURE_PATH

_C.base_dir_list = [
    # '20240708', #done small size
    
    '20240920', #done, copied to NT
    '20240930', #done, copied to NT
    '20241001', #done, copied to NT
    '20241020', #done, copied to NT
    '20241121', #done, copied to NT
    '20241210', #done, copied to NT
    '20241228', #done, copied to NT
    '20250111', #done, copied to NT
    '20250331', #done, copied to NT
    '20250727', #done, copied to NT
    '20250914', #done, , copied to NT
    
    # '20251022-demo'
    
    # 'DatasetCell'
    
    # 'RGB_segmented',
    # 'Original_images'
              
]

_C.DATA.SEED = 1337
# _C.DATA.SEED = 42
# _C.DATA.SEED = 2024

# _C.DATA.SEEDS = [
    # 1337,
    # 42,
    # 2024
    # ]

_C.DATA.C = 3
_C.DATA.W = _C.DATA.H = 256
_C.DATA.IMAGE_SIZE = _C.DATA.W
_C.DATA.DIMENSION = _C.DATA.W, _C.DATA.H, _C.DATA.C
_C.DATA.TEST_SIZE = 0.3
_C.DATA.CLASS_LABELS = ['IDA', 'THL']
_C.DATA.CLASSES = [0, 1]
_C.DATA.CLASSDICT = [(0, "IDA"), (1, "THL")]
_C.DATA.val_ratio = 0.3


_C.DATA.COLOR = 'BGR' #just for cross-check, not use to code
# _C.DATA.COLOR = 'RGB'
# _C.DATA.COLOR = 'he'
# _C.DATA.COLOR = 'hsv'
# _C.DATA.COLOR = 'lab'

_C.MODEL = CN()
_C.MODEL.SAVEPATH = os.path.join(_C.BASEPATH, 'models')

# if _C.DATA.W == 256:
#     _C.MODEL.NAMES = [
#         # 'ViT_b16',
#         # 'ViT_l16',
#         # 'ViT_b32',
#         # 'vit_l32'
#         ]
    
# if _C.DATA.W == 256:    
_C.MODEL.NAMES = [
    # 'AlexNet',
    # 'ViT_b16',       
    # 'vit_l32',
    # 'VGG16',
    # 'VGG19',                      
    # 'DenseNet121',
    # 'DenseNet201',
    # 'MobileNetV2',      
    # 'InceptionResNetV2',  

    # 'MobileNet', 
               
    # 'InceptionV3',      
    # 'EfficientNetB7',
    # 'EfficientNetV2B3',
               
    # 'EfficientNetV2S',
    # 'ViT_b32',
    # 'ViT_l16',
    # 'EfficientNetV2M',
    
    # 'EfficientNetV2L',       
    # 'ConvNeXtTiny',
    # 'ConvNeXtSmall',
    # 'ConvNeXtBase',
    'ConvNeXtLarge',
    # 'ConvNeXtXLarge', 
    
    # 'ViT_b16',
    
    # 'ResNet50',
    # 'ResNet50V2',
    # 'ViT_b32', 
    # 'Xception',
]   
    
# if _C.DATA.W == 227:  
#     _C.MODEL.NAMES = [
#         'AlexNet'
#         ]

# _C.MODEL.INFER_FILE_PATH = 'infer-ens-20250302-12THL.csv'
# _C.MODEL.EVA_FILE_PATH = 'Eva-ens-20250302'

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
# Before starting Phase 3 training loop:
_C.TRAIN.reduceLR_patience = 8   # Increase from 2 to 8
_C.TRAIN.stop_patience = 20      # Increase from 12 to 20
_C.DATA.N_SPLIT = 5
_C.TRAIN.Actual_Fold = 4 #the last is N_SPLIT-1
_C.TRAIN.Feature_Layer_Index = -19 #batch_normalization_12 (None, 1536) True
_C.TRAIN.SAMPLE = False

if platform.system() == 'Darwin' or platform.system() == 'Windows':    
    _C.TRAIN.verbose = 1
    
if platform.system() == 'Linux':    
    _C.TRAIN.verbose = 2

# _C.TRAIN.LR_list = [
#                 # 1e-5,
#                 # 1e-4,
#                 # 1e-3,
#                 # 1e-2,
#                 # 1e-1,
#                 ]

# For ViT base
_C.TRAIN.ACTIVATION = 'gelu'

_C.DATA.VERIFY = False
# _C.DATA.VERIFY = True

# _C.TRAIN.SAMPLE = True
if _C.TRAIN.SAMPLE and platform.system() == 'Windows':
    # _C.TRAIN.SAMPLE_SIZE = 1024
    # _C.TRAIN.SAMPLE_SIZE_VAL = int(_C.TRAIN.SAMPLE_SIZE * _C.DATA.val_ratio)
    # _C.TRAIN.SAMPLE_SIZE_TEST = 30
    # _C.TRAIN.BATCH_SIZE = 32
    # _C.TRAIN.EPOCHS = 60
    # _C.DATA.N_SPLIT = 3
    # _C.TRAIN.Actual_Fold = 0
    # _C.TRAIN.verbose = 1

    _C.TRAIN.SAMPLE_SIZE = 128
    _C.TRAIN.SAMPLE_SIZE_VAL = int(_C.TRAIN.SAMPLE_SIZE * _C.DATA.val_ratio)
    _C.TRAIN.SAMPLE_SIZE_TEST = 32 #patch
    _C.TRAIN.BATCH_SIZE = 8
    _C.TRAIN.EPOCHS = 2
    _C.DATA.N_SPLIT = 5
    _C.TRAIN.Actual_Fold = 4
    _C.TRAIN.verbose = 1
    
if _C.TRAIN.SAMPLE and platform.system() == 'Linux':
    _C.TRAIN.SAMPLE_SIZE = 1024
    _C.TRAIN.SAMPLE_SIZE_VAL = int(_C.TRAIN.SAMPLE_SIZE * _C.DATA.val_ratio)
    _C.TRAIN.SAMPLE_SIZE_TEST = 256
    _C.TRAIN.BATCH_SIZE = 32
    _C.TRAIN.EPOCHS = 60
    _C.DATA.N_SPLIT = 3
    _C.TRAIN.Actual_Fold = 2
    _C.TRAIN.verbose = 1

_C.TRAIN.Enable = False  #Phase 1 & 2
# _C.TRAIN.Enable = True #Phase 1 & 2
if  _C.TRAIN.Enable:
    _C.PILOT = False
else:
    _C.PILOT = True

_C.TRAIN.Phase3 = False
# _C.TRAIN.Phase3 = True

# _C.Evaluate_Val = True  
_C.Evaluate_Val = False

_C.find_ood = False
_C.find_threshold = False  

# _C.PLOT_CURVATURE = False
_C.PLOT_CURVATURE = False

_C.Evaluate_Test = False
_C.n_select = 3
# _C.Evaluate_Test = False
_C.External_Infer = False
_C.Performance_Report = False
_C.Inference = True
_C.Inference_Audit = False
_C.Inference_Report = False
_C.External_Patch_Audit = False
_C.Performance_Comparison = False
_C.Ablation_Study = False

# _C.SAVE = True
_C.SAVE = False
# _C.SAVEHIST = True

#old
#T_LOWER, T_UPPER, T_OPT = 0.5130, 0.5729, 0.5538

# compute_hybrid_thresholds
_C.T_LOWER, _C.T_UPPER, _C.T_OPT = 0.4914, 0.6505, 0.5742

    
def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    #update_config(config, args)

    return config

