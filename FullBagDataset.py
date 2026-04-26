# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 09:36:18 2025

@author: Science
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.utils import to_categorical

# class PatientBagDataset(PatchDataset):
class PatientBagDataset(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=4, K=32,
                 preprocessing_function=None,
                 shuffle=True,
                 fp_ids=None, fn_ids=None,
                 weight_fp=2.0, weight_fn=3.0, weight_normal=1.0,
                 mode='train'):

        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.K = K
        self.preprocessing_function = preprocessing_function,
        self.shuffle = shuffle
        self.mode = mode

        # Hard patient sets
        self.fp_ids = set(fp_ids) if fp_ids else set()
        self.fn_ids = set(fn_ids) if fn_ids else set()

        # Weight configuration
        self.weight_fp = weight_fp
        self.weight_fn = weight_fn
        self.weight_normal = weight_normal

        # Precompute patient groups
        grouped = df.groupby('patient_id')
        self.patients = list(grouped.groups.keys())
        self.patient_groups = {pid: grouped.get_group(pid) for pid in self.patients}
        self.labels = {pid: grouped.get_group(pid)['label'].iloc[0] for pid in self.patients}
        
        self.on_epoch_end()
    
    @property
    def classes(self):
        return np.array([self.labels[pid] for pid in self.patients])

    def __len__(self):
        return int(np.ceil(len(self.patients) / self.batch_size))


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.patients)


    def _load_patch(self, patch_path):
        img_ori = cv2.imread(str(patch_path), cv2.IMREAD_UNCHANGED)
             
        if img_ori is None:
            raise ValueError(f"Failed to load image: {patch_path}")

        # --- Handle color modes -> Ensure RGB uint8 [0, 255] ---
        if len(img_ori.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img_ori, cv2.COLOR_GRAY2RGB)
        elif img_ori.shape[-1] == 4:  # BGRA/RGBA -> RGB
            # Assuming input might be BGRA from cv2, convert to RGBA then RGB
            img_rgba = cv2.cvtColor(img_ori, cv2.COLOR_BGRA2RGBA)
            img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
        elif img_ori.shape[-1] == 3:  # BGR -> RGB
             img_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unexpected image shape: {img_ori.shape} at {patch_path}")

        # --- Convert to Float [0, 1] ---
        img_rgb_float = img_rgb.astype(np.float32) / 255.0
        return img_rgb_float # Return Numpy array [0,1] range
    
    def _apply_model_preprocessing(self, img_bgr_uint8):
        """
        img_bgr_uint8: NumPy uint8 image in BGR order
        Returns float32 preprocessed image.
        """
    
        # Ensure preprocessing_function is callable
        if callable(self.preprocessing_function):
            try:
                img_processed = self.preprocessing_function(img_bgr_uint8)
            except Exception as e:
                raise RuntimeError(f"Error in preprocessing_function: {e}")
            return img_processed.astype(np.float32)
    
        # If no preprocessing, fallback to [0,1] float
        return (img_bgr_uint8.astype(np.float32) / 255.0)

    def _get_patient_weight(self, pid):
        if pid in self.fp_ids:
            return self.weight_fp
        elif pid in self.fn_ids:
            return self.weight_fn
        else:
            return self.weight_normal


    def __getitem__(self, idx, **kwargs):
        """
        Keras-compatible output:
          • train: returns (X, y_onehot, sample_weight)
          • val:   returns (X, y_onehot)
          • test:  returns X
        Patient IDs are stored internally for later retrieval.
        """
        get_pids = kwargs.get("get_pids", False)
        # ---- compute indices ----
        start = idx * self.batch_size
        end   = min(start + self.batch_size, len(self.patients))
        batch_patients = self.patients[start:end]
    
        # Store patient_ids for later inspection (e.g. after prediction)
        self.last_batch_ids = batch_patients
    
        bag_batch, label_batch, weight_batch = [], [], []
    
        for pid in batch_patients:
    
            patch_paths = self.patient_groups[pid]['patch_path'].tolist()
            num_patches = len(patch_paths)
    
            # ---- sample/pad to K patches ----
            if num_patches >= self.K:
                chosen = np.random.choice(patch_paths, self.K, replace=False).tolist()
            else:
                repeats = -(-self.K // num_patches)
                chosen = (patch_paths * repeats)[:self.K]
    
            imgs = np.stack([self._load_patch(p) for p in chosen])
            bag_batch.append(imgs)
    
            label_batch.append(self.labels[pid])
            weight_batch.append(self._get_patient_weight(pid))
    
        # ---- construct arrays ----
        X = np.stack(bag_batch)                         # (B, K, H, W, C)
        y = np.array(label_batch, dtype=np.int32)       # (B,)
        w = np.array(weight_batch, dtype=np.float32)    # (B,)
        y_onehot = to_categorical(y, num_classes=2)     # (B, 2)

        if self.preprocessing_function is not None:
            B, K, H, W, C = X.shape
            X_processed = np.zeros((B, K, H, W, C), dtype=np.float32)
        
            for b in range(B):
                for k in range(K):
                    patch_uint8 = (X[b, k] * 255).astype(np.uint8)
                    patch_bgr   = cv2.cvtColor(patch_uint8, cv2.COLOR_RGB2BGR)
        
                    processed = self._apply_model_preprocessing(patch_bgr)
        
                    # Ensure the shape matches
                    if processed.shape != (H, W, C):
                        raise ValueError(
                            f"Preprocessing returned wrong shape: "
                            f"{processed.shape} for expected {(H, W, C)}"
                        )
        
                    X_processed[b, k] = processed
        
            X = X_processed
 
        # ---- output depending on mode ----
        if self.mode == 'train':
            if get_pids:
                return batch_patients, X, y_onehot, w                # Keras expects exactly (x, y, w)
            else:
                return  X, y_onehot, w
        elif self.mode == 'val':
            if get_pids:
                return batch_patients, X, y_onehot
            else:
                return X, y_onehot                 # No sample weights
        else:
            return X                            # Prediction-only mode


    def get_batch_with_ids(self, idx):
        """
        Return (patient_ids, X, y_onehot, sample_weights) for debugging/plotting.
        Mirrors previous behavior but is explicit.
        """
        batch_idx = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_idx = batch_idx[batch_idx < len(self.patients)]
        batch_patients = [self.patients[i] for i in batch_idx]

        # reuse __getitem__ logic but bypass Keras-compatible output
        # (call internal routines to avoid duplication)
        bag_batch, label_batch, weight_batch = [], [], []
        for pid in batch_patients:
            patch_paths = self.patient_groups[pid]['patch_path'].tolist()
            if len(patch_paths) >= self.K:
                chosen = np.random.choice(patch_paths, self.K, replace=False).tolist()
            else:
                repeats = -(-self.K // len(patch_paths))
                chosen = (patch_paths * repeats)[:self.K]
            imgs = np.stack([self._load_patch(p) for p in chosen])
            bag_batch.append(imgs)
            label_batch.append(self.labels[pid])
            weight_batch.append(self._get_patient_weight(pid))

        X = np.stack(bag_batch)
        y_onehot = to_categorical(np.array(label_batch, dtype=np.int32), num_classes=2)
        w = np.array(weight_batch, dtype=np.float32)
        return batch_patients, X, y_onehot, w

    def plot_random_patient_batch(self, idx=None, patient_to_plot=4, save=False):
        """
        Visualize patient-level bags from a specific batch index.
    
        Args:
            idx (int or None): Batch index. If None → random batch.
            patient_to_plot (int): Max number of patients to plot from the batch.
            save (bool): Save each figure.
        """
    
        # ----- Select batch index -----
        if idx is None:
            idx = np.random.randint(0, len(self))
    
        print(f"\n[PLOT] Using batch index: {idx}")
    
        # # ----- Fetch batch in appropriate mode -----
        # if self.mode == 'train':
        #     batch_patients, images, labels, w_batch = self.get_batch_with_ids(idx)
        # elif self.mode=='val':
        #     batch_patients, images, labels = self.get_batch_with_ids(idx)
        #     w_batch = None
        
        batch_patients, images, labels, w_batch = self.get_batch_with_ids(idx)   
        B, K, H, W, C = images.shape
        print(f"[PLOT] Batch shape = (B={B}, K={K}, H={H}, W={W}, C={C})")
    
        # How many patients to visualize?
        n_plot = min(B, patient_to_plot)
        print(f"{n_plot=}")
    
        for i in range(n_plot):
    
            pid = batch_patients[i]
            label = labels[i]
    
            # ----------------------------
            # FIX: Prevent duplicate patches in visualization
            # ----------------------------
            unique_paths = self.patient_groups[pid]["patch_path"].unique().tolist()
            num_unique = len(unique_paths)
    
            if num_unique >= K:
                chosen_paths = np.random.choice(unique_paths, K, replace=False).tolist()
            else:
                chosen_paths = unique_paths  # fewer than K → show only unique patches
    
            K_show = len(chosen_paths)
    
            # ----------------------------
            # Build grid layout  (default for K=32 → 8×4)
            # ----------------------------
            if K_show == 32:
                nrows, ncols = 4, 8
            else:
                ncols = int(np.ceil(np.sqrt(K_show)))
                nrows = int(np.ceil(K_show / ncols))
    
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                     figsize=(12, 10), dpi=150)
            axes = np.array(axes).ravel()
    
            # FP / FN marking
            if pid in self.fp_ids:
                status = " (FP)"
            elif pid in self.fn_ids:
                status = " (FN)"
            else:
                status = ""
    
            # ----------------------------
            # Populate grid with images
            # ----------------------------
            for j, ax in enumerate(axes):
                ax.set_xticks([]); ax.set_yticks([])
    
                if j < K_show:
                    img = self._load_patch(chosen_paths[j])
                    ax.imshow(img)
                else:
                    ax.axis("off")
    
                # frame
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color("lightgray")
                    s.set_linewidth(0.8)
    
            # ----------------------------
            # Title and layout
            # ----------------------------
            title = f"Patient {pid} | Label={label}{status}"
            fig.text(0.5, 0.02, title, ha="center",
                     fontsize=14, weight='bold')
    
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.show()
    
            # ----------------------------
            # Save
            # ----------------------------
            if save:
                safe = pid.replace("/", "_")
                fname = f"patientbag_{safe}.png"
                fig.savefig(fname)
                print(f"  Saved → {fname}")

class FullBagDataset(tf.keras.utils.Sequence):
    """
    Full-bag generator for inference and feature extraction.
    Logic strictly aligned with PatchDatasetPreserve for color consistency.
    """

    def __init__(self, 
                 config, 
                 df, 
                 preprocessing_function=None, 
                 shuffle=False, 
                 expect_rgba=True, 
                 bg_color=(0, 0, 0),
                 mode='test' 
                 ):
        
        self.config = config
        self.df = df
        self.preprocessing_function = preprocessing_function
        self.shuffle = shuffle
        self.expect_rgba = expect_rgba
        self.bg_color = bg_color
        self.batch_size = 1  
        self.mode = mode

        # Group by patient
        grouped = df.groupby('patient_id')
        self.patients = list(grouped.groups.keys())
        self.patient_groups = {pid: grouped.get_group(pid) for pid in self.patients}
        
        if 'label' in df.columns:
            self.labels = {pid: grouped.get_group(pid)['label'].iloc[0] for pid in self.patients}
        else:
            self.labels = {pid: 0 for pid in self.patients}

        print(f"[FullBagDataset] Initialized in '{self.mode}' mode.")
        print(f" - Patients: {len(self.patients)}")
        print(f" - Expect RGBA: {self.expect_rgba}")

    def __len__(self):
        return int(np.ceil(len(self.patients) / self.batch_size))
    
    @property
    def classes(self):
        return np.array([self.labels[pid] for pid in self.patients])

    def _load_and_preprocess_single_patch(self, patch_path):
        """
        Loads a single image.
        
        CRITICAL ALIGNMENT FIX:
        - If expect_rgba=True: Returns BGR (Matches PatchDatasetPreserve logic).
        - If expect_rgba=False: Returns RGB (Matches PatchDatasetPreserve logic).
        """
        full_path = str(patch_path)
        
        # --- 1. Load Image ---
        if self.expect_rgba:
            # Logic mirrors PatchDatasetPreserve._load_and_preprocess_image
            # Load as RGBA (OpenCV defaults to BGRA)
            img_ori = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img_ori is None:
                raise ValueError(f"Failed to load image: {full_path}")

            if img_ori.shape[2] == 4:
                # Extract Channels. img_ori is BGRA.
                # slicing [..., :3] gives BGR.
                rgb = img_ori[..., :3].astype(np.float32) 
                alpha = img_ori[..., 3].astype(np.float32) / 255.0
                alpha = np.stack([alpha]*3, axis=-1)
                
                bg = np.array(self.bg_color, dtype=np.float32)
                
                # Blend: (Source * Alpha) + (Background * (1 - Alpha))
                # The result `new_rgb` is actually BGR because source `rgb` was BGR.
                new_rgb = rgb * alpha + bg * (1 - alpha)
                img_processed = new_rgb.astype(np.uint8)
                
                # NOTE: PatchDatasetPreserve returns here WITHOUT converting to RGB.
                # We must do the same to match the model's training input.
            
            elif img_ori.shape[2] == 3:
                # Fallback: Image is RGB/BGR (no alpha). OpenCV loads as BGR.
                img_processed = img_ori 
            else:
                 raise ValueError(f"Unexpected channel count {img_ori.shape} for RGBA mode")

        else:
            # Logic mirrors PatchDatasetPreserve._load_and_preprocess_image_rgb
            # Load as standard color (BGR)
            img_ori = cv2.imread(full_path, cv2.IMREAD_COLOR) 
            if img_ori is None:
                raise ValueError(f"Failed to load image: {full_path}")
            
            # PatchDatasetPreserve explicitly converts to RGB here.
            img_processed = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        # --- 2. Resize ---
        if img_processed.shape[0] != 256:
             img_processed = cv2.resize(img_processed, (256, 256), interpolation=cv2.INTER_AREA)

        # --- 3. Preprocessing ---
        # img_processed is BGR if expect_rgba=True, RGB if False.
        if self.preprocessing_function:
            img_float = img_processed.astype(np.float32)
            processed = self.preprocessing_function(img_float)
            
            # Normalize to [0, 1] if max > 1.0 (Safety check)
            return processed / 255.0 if np.max(processed) > 1.0 else processed
        else:
            # Default scaling to [0, 1]
            return img_processed.astype(np.float32) / 255.0

    def __getitem__(self, idx, **kwargs):
            get_pids = kwargs.get("get_pids", False)
           
            start = idx * self.batch_size
            end = min(start + self.batch_size, len(self.patients))
            batch_patients = self.patients[start:end]
            
            if not batch_patients:
                return np.array([]), np.array([])
    
            pid = batch_patients[0]
            patch_paths = self.patient_groups[pid]["patch_path"].tolist()
            
            root = Path(self.config.DATASET)

            processed_imgs = []
            
            for p in patch_paths:
                img_path = root / p
                try:
                    img = self._load_and_preprocess_single_patch(img_path)
                    processed_imgs.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")
                    continue
            
            if not processed_imgs:
                print(f"Warning: No valid images for patient {pid}")
                return np.array([]), np.array([])

            # Stack -> (N, H, W, C)
            X_patient = np.stack(processed_imgs)
            # Add batch dim -> (1, N, H, W, C)
            X_batch = np.expand_dims(X_patient, axis=0)
    
            y = np.array([self.labels[pid]], dtype=np.int32)
            y_batch = to_categorical(y, num_classes=2)
    
            if self.mode == 'external':
                return batch_patients, X_batch
            elif self.mode == 'test':
                if get_pids: return batch_patients, y, X_batch, y_batch
                return X_batch, y_batch
            elif self.mode == 'inference':
                if get_pids: return batch_patients, X_batch, y_batch
                return X_batch, y_batch
            
            return X_batch, y_batch
    
    def plot_random_patient_batch(self, idx=None, patient_to_plot=1, save_dir=None):
        """
        Visualizes the actual tensor data. 
        If expect_rgba=True, images are BGR. We convert to RGB here ONLY for display 
        so the human sees them correctly, confirming the data is valid BGR.
        """
        if idx is None:
            idx = np.random.randint(0, len(self))
            
        print(f"\n[PLOT] Starting at batch index: {idx}")

        for i in range(patient_to_plot):
            current_idx = idx + i
            if current_idx >= len(self): break
            
            try:
                res = self.__getitem__(current_idx, get_pids=True)
                # Unpack based on mode (simplified for brevity)
                if self.mode == 'external': batch_patients, X_batch = res; y_batch = np.zeros((1,2))
                elif self.mode == 'test': batch_patients, _, X_batch, y_batch = res
                else: batch_patients, X_batch, y_batch = res
            except Exception as e:
                print(f"[PLOT] Error: {e}")
                continue

            if len(batch_patients) == 0: continue

            pid = batch_patients[0]
            label = np.argmax(y_batch[0])
            bag_images = X_batch[0]
            
            # Sampling for display
            indices = np.random.choice(bag_images.shape[0], min(32, bag_images.shape[0]), replace=False)
            display_images = bag_images[indices]
            
            # Plotting Setup
            n_show = len(display_images)
            ncols = 8
            nrows = int(np.ceil(n_show / ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 2.5 * nrows), dpi=150)
            if nrows * ncols == 1: axes = np.array([axes])
            axes = axes.ravel()

            for j, ax in enumerate(axes):
                ax.set_xticks([]); ax.set_yticks([])
                if j < n_show:
                    img_vis = display_images[j].copy()
                    
                    # --- Logic for DISPLAY ONLY ---
                    # If we are in the RGBA mode (BGR data), we must swap channels 
                    # so the plot looks Red (Correct), proving the data is BGR.
                    if self.expect_rgba:
                        # Check if data is normalized (approx range)
                        if img_vis.max() <= 1.0:
                            img_vis = (img_vis * 255).astype(np.uint8)
                            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
                            img_vis = img_vis.astype(np.float32) / 255.0
                        else:
                            # Standard uint8 handling
                            img_vis = cv2.cvtColor(img_vis.astype(np.uint8), cv2.COLOR_BGR2RGB)

                    # --- Safety Norm ---
                    if img_vis.min() < 0: img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-7)
                    elif img_vis.max() > 1.0: img_vis = img_vis / 255.0
                    
                    ax.imshow(np.clip(img_vis, 0, 1))
                    for spine in ax.spines.values(): spine.set_color('lightgray')
                else:
                    ax.axis('off')

            fig.suptitle(f"Patient: {pid} | Label: {label} | Mode: {self.mode}\n(Input is BGR? {self.expect_rgba} -> Converted to RGB for Display)", fontsize=16, y=1.05)
            plt.tight_layout()
            plt.show()
            
            if save_dir:
                safe_pid = str(pid).replace("/", "_")
                fname = os.path.join(self.config.SAVE_PATH, save_dir, f"patient_{safe_pid}.png")
                fig.savefig(fname, bbox_inches='tight')
                print(f"Save => {fname}")
                
class FullBagDataset_backup(tf.keras.utils.Sequence):
    """
    Full-bag generator for inference and feature extraction.
    
    Features:
    - Loads *all* patches for a patient (no sampling/padding).
    - Robust Image Loading: Handles RGBA (transparency) and RGB.
    - Consistent Preprocessing: Matches PatchDatasetPreserve logic.
    - Output: Varies based on 'mode' (test, inference, external).
    """

    def __init__(self, 
                 config, 
                 df, 
                 preprocessing_function=None, 
                 shuffle=False, 
                 expect_rgba=True, 
                 bg_color=(0, 0, 0),
                 mode='test' # Options: 'test', 'inference', 'external'
                 ):
        
        self.config = config
        self.df = df
        self.preprocessing_function = preprocessing_function
        self.shuffle = shuffle
        self.expect_rgba = expect_rgba
        self.bg_color = bg_color
        self.batch_size = 1  # Always processing one patient at a time
        self.mode = mode

        # Group by patient
        grouped = df.groupby('patient_id')
        self.patients = list(grouped.groups.keys())
        self.patient_groups = {pid: grouped.get_group(pid) for pid in self.patients}
        
        # Store labels if available, else default to 0
        if 'label' in df.columns:
            self.labels = {pid: grouped.get_group(pid)['label'].iloc[0] for pid in self.patients}
        else:
            self.labels = {pid: 0 for pid in self.patients}

        print(f"[FullBagDataset] Initialized in '{self.mode}' mode.")
        print(f" - Patients: {len(self.patients)}")
        print(f" - Expect RGBA: {self.expect_rgba}")
        print(f" - Preprocessing: {self.preprocessing_function}")

    def __len__(self):
        return int(np.ceil(len(self.patients) / self.batch_size))
    
    @property
    def classes(self):
        return np.array([self.labels[pid] for pid in self.patients])

    def _load_and_preprocess_single_patch(self, patch_path):
            """
            Loads a single image.
            CRITICAL FIX: Aligns color space logic with PatchDatasetPreserve.
            - If expect_rgba=True: Returns BGR (Matches training loader logic).
            - If expect_rgba=False: Returns RGB.
            """
            full_path = str(patch_path)
            
            # --- 1. Load Image (OpenCV loads in BGR by default) ---
            if self.expect_rgba:
                # Load as RGBA (BGR + Alpha)
                img_ori = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                if img_ori is None:
                    raise ValueError(f"Failed to load image: {full_path}")
    
                if img_ori.shape[2] == 4:
                    # Extract Channels (BGR) and Alpha
                    bgr = img_ori[..., :3].astype(np.float32) # BGR
                    alpha = img_ori[..., 3].astype(np.float32) / 255.0
                    alpha = np.stack([alpha]*3, axis=-1)
                    
                    bg = np.array(self.bg_color, dtype=np.float32)
                    
                    # Blend: (Source * Alpha) + (Background * (1 - Alpha))
                    # Result is still BGR
                    new_bgr = bgr * alpha + bg * (1 - alpha)
                    img_final = new_bgr.astype(np.uint8)
                
                elif img_ori.shape[2] == 3:
                    img_final = img_ori # Already BGR
                else:
                     raise ValueError(f"Unexpected channel count {img_ori.shape}")
                     
                # --- ALIGNMENT FIX: Do NOT convert to RGB for RGBA mode ---
                # PatchDatasetPreserve returns BGR here. We must match it.
                img_processed = img_final 
    
            else:
                # Load as RGB (Standard path)
                img_ori = cv2.imread(full_path, cv2.IMREAD_COLOR) # BGR
                if img_ori is None:
                    raise ValueError(f"Failed to load image: {full_path}")
                
                # --- ALIGNMENT FIX: Convert to RGB for non-RGBA mode ---
                # PatchDatasetPreserve converts to RGB here. We must match it.
                img_processed = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    
            # --- 2. Resize ---
            if img_processed.shape[0] != 256:
                 img_processed = cv2.resize(img_processed, (256, 256), interpolation=cv2.INTER_AREA)
    
            # --- 3. Preprocessing ---
            if self.preprocessing_function:
                img_float = img_processed.astype(np.float32)
                processed = self.preprocessing_function(img_float)
                
                # Normalize to [0, 1] if needed (heuristic)
                return processed / 255.0 if np.max(processed) > 1.0 else processed
            else:
                # Default scaling
                return img_processed.astype(np.float32) / 255.0

    def __getitem__(self, idx, **kwargs):
            """
            Returns items based on 'mode' and 'get_pids'.
            
            mode='test':
               - get_pids=True:  (batch_patients, labels, X_batch, y_batch)
               - get_pids=False: (X_batch, y_batch)
               
            mode='inference':
               - get_pids=True:  (batch_patients, X_batch, y_batch)
               - get_pids=False: (X_batch, y_batch)
               
            mode='external':
               - returns: (batch_patients, X_batch)
            """
            get_pids = kwargs.get("get_pids", False)
            mode = kwargs.get("mode", 'test')
    
            # Select patient
            start = idx * self.batch_size
            end = min(start + self.batch_size, len(self.patients))
            batch_patients = self.patients[start:end]
            
            if not batch_patients:
                return np.array([]), np.array([])
    
            pid = batch_patients[0]
            # Retrieve patch paths from DataFrame
            patch_paths = self.patient_groups[pid]["patch_path"].tolist()
    
            # Load all patches for this patient
            root = Path(self.config.DATASET)
            processed_imgs = []
            
            for p in patch_paths:
                # Combine root + relative path
                img_path = root / p
                try:
                    img = self._load_and_preprocess_single_patch(img_path)
                    processed_imgs.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")
                    continue
            
            if not processed_imgs:
                # Return empty batch if all images fail
                print(f"Warning: No valid images for patient {pid}")
                return np.array([]), np.array([])

            # Stack into (N_patches, H, W, C)
            X_patient = np.stack(processed_imgs)
            
            # Add batch dimension -> (1, N_patches, H, W, C)
            X_batch = np.expand_dims(X_patient, axis=0)
    
            # Labels (Integer and One-Hot)
            y = np.array([self.labels[pid]], dtype=np.int32)
            y_batch = to_categorical(y, num_classes=2)
    
            # --- Return Logic based on Mode ---
            
            if mode == 'external':
                # External Mode: Just ID and Data (No labels assumed, or ignored)
                return batch_patients, X_batch
    
            elif mode == 'test':
                # Test Mode: Needs full label info for evaluation
                if get_pids:
                    return batch_patients, y, X_batch, y_batch
                return X_batch, y_batch
    
            elif mode == 'inference':
                # Inference Mode: Standard output structure for pipeline
                if get_pids:
                    return batch_patients, X_batch, y_batch
                return X_batch, y_batch
            
            # Default Fallback (Standard Keras)
            return X_batch, y_batch
    
    def plot_random_patient_batch(self, idx=None, patient_to_plot=1, save_dir=None):
        """
        Visualize patient-level bags by retrieving data via __getitem__.
        This ensures the plot shows the exact preprocessed tensors the model receives.
        
        Args:
            idx (int or None): Batch index. If None, picks random.
            patient_to_plot (int): Number of consecutive patients to plot.
            save (bool): If True, saves the figure.
        """
        # 1. Select starting index
        if idx is None:
            idx = np.random.randint(0, len(self))
            
        print(f"\n[PLOT] Starting at batch index: {idx}")

        # 2. Loop over requested number of patients
        for i in range(patient_to_plot):
            current_idx = idx + i
            if current_idx >= len(self):
                break
            
            # --- Fetch Data directly from __getitem__ ---
            try:
                # We assume test/inference mode return signature for visualization
                res = self.__getitem__(current_idx, get_pids=True)
                
                if self.mode == 'external':
                    batch_patients, X_batch = res
                    y_batch = np.zeros((1, 2)) # Dummy label
                elif self.mode == 'test':
                    batch_patients, _, X_batch, y_batch = res
                else:
                    batch_patients, X_batch, y_batch = res
                    
            except Exception as e:
                print(f"[PLOT] Error fetching batch {current_idx}: {e}")
                continue

            if len(batch_patients) == 0:
                print(f"[PLOT] Batch {current_idx} is empty.")
                continue

            # Unpack single patient from batch
            pid = batch_patients[0]
            label_vec = y_batch[0]
            label = np.argmax(label_vec) # Convert one-hot to index (0 or 1)
            
            bag_images = X_batch[0]      # Remove batch dim -> (N, H, W, C)
            total_patches = bag_images.shape[0]

            # 3. Randomly Sample Max 32 Patches
            max_patches = 32
            if total_patches > max_patches:
                indices = np.random.choice(total_patches, max_patches, replace=False)
                display_images = bag_images[indices]
            else:
                display_images = bag_images
            
            n_show = len(display_images)
            
            # 4. Define Layout (8 columns fixed)
            ncols = 8
            nrows = int(np.ceil(n_show / ncols))
            
            # Adjust figure height: approx 2.5 inches per row
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                                     figsize=(16, 2.5 * nrows), dpi=150)
            
            # Handle single subplot case
            if nrows * ncols == 1:
                axes = np.array([axes])
            axes = axes.ravel()

            # 5. Populate Grid
            for j, ax in enumerate(axes):
                ax.set_xticks([])
                ax.set_yticks([])
                
                if j < n_show:
                    img_tensor = display_images[j] # Shape (H, W, C)
                    
                    # --- De-processing for Visualization ---
                    img_vis = img_tensor.copy()
                    
                    # If min < 0 (e.g., tf.keras.applications.resnet50.preprocess_input), shift it
                    if img_vis.min() < 0:
                        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-7)
                    
                    # If max > 1 (e.g., 0-255 range), scale it
                    elif img_vis.max() > 1.0:
                        img_vis = img_vis / 255.0
                    
                    # Clip to be safe
                    img_vis = np.clip(img_vis, 0, 1)

                    ax.imshow(img_vis)
                    
                    # Add border
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color('lightgray')
                else:
                    ax.axis('off')

            # 6. Title
            fig.suptitle(f"Patient: {pid} | Label: {label}\n"
                         f"Showing {n_show}/{total_patches} patches (Preprocessed Input)", 
                         fontsize=16, y=0.98 if nrows > 1 else 1.05)
            
            plt.tight_layout()
            plt.show()

            if save_dir is not None:
                safe_pid = str(pid).replace("/", "_")
                fname = os.path.join(self.config.SAVE_PATH, save_dir, f"patient_{safe_pid}_preprocessed.png")
                fig.savefig(fname, bbox_inches='tight')
                print(f"[Saved] {fname}")

#%%