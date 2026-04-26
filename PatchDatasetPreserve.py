import tensorflow as tf
import os
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from pathlib import Path
from PatchDataset import PatchDataset
# class PatchDatasetPreserve(tf.keras.utils.Sequence):
class PatchDatasetPreserve(PatchDataset):
    def __init__(self,
                 config,
                 df,
                 batch_size=32,
                 preprocessing_function=None,
                 shuffle=True,
                 use_cutmix=False,
                 cutmix_alpha=1.0,
                 cutmix_version='v1',
                 expect_rgba=True,   # NEW: allow RGB mode
                 ):
        self.config = config
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.preprocessing_function = preprocessing_function
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_version = cutmix_version
        self.expect_rgba = expect_rgba   # NEW FLAG

        print("PatchDatasetPreserve (cell shape and size) initialized.")
        print(f"Expect RGBA: {self.expect_rgba}")
        print(f"Preprocessing function: {self.preprocessing_function}")
        print(f"{self.config.DATA.COLOR=}")
        self.on_epoch_end()


    @property
    def classes(self):
        return self.df['label'].to_numpy()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)  
    
    def _load_and_preprocess_image_rgb(self, row):
        root = Path(self.config.DATASET) 
        img_path = row['patch_path'] 
        full_path = root / img_path
        img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)  # BGR

        if img is None:
            raise ValueError(f"Unable to load image: {img_path}")
       
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        if img.shape[0] != 256:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32)

        # Apply preprocessing if any
        if self.preprocessing_function is not None:
            img = self.preprocessing_function(img)
            return img / 255.0
        else:
            return img / 255.0
        
    # On Windows, this outputs:  H:\My Drive\THL-g\20240920\0\...\cell.png
    # On Linux, this outputs:    /mnt/data/THL-g/20240920/0/.../cell.png
    
    def _load_and_preprocess_image(self, row,
                                    bg_color=(0,0,0)
                                   # bg_color=(245,245,245)
                                   ):
        # white = (255,255,255)
        # black = (0,0,0)
        # gray = (245,245,245)
        root = Path(self.config.DATASET) 
        img_path = row['patch_path'] 
        full_path = root / img_path      
        # 1. Debug: Check if file exists
        if not os.path.exists(str(full_path)):
            raise FileNotFoundError(f"File not found on disk: {img_path}")
            
        rgba = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise ValueError(f"cv2.imread failed (returned None). Corrupt or invalid file: {img_path}")
            
        # print(f"{rgba.shape=}")
        if rgba.shape[2] != 4:
            raise ValueError("Image is not RGBA.")
    
        rgb = rgba[..., :3].astype(np.float32)
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        alpha = np.stack([alpha]*3, axis=-1)
    
        bg = np.array(bg_color, dtype=np.float32)
        
        # Blend: (Source * Alpha) + (Background * (1 - Alpha))
        new_rgb = rgb * alpha + bg * (1-alpha)
        new_rgb = new_rgb.astype(np.uint8)
        
        if self.preprocessing_function is not None:
            img_rgb_float = new_rgb.astype(np.float32)
            # (ConvNeXt/EffNetV2) often handle standard RGB float.
            # -> float32 [0,255]
            img_final = self.preprocessing_function(img_rgb_float)
            # print(np.min(img_final), np.max(img_final)) #0.0 199.0
            return img_final/255.0
        else:
            img_final = new_rgb.astype(np.float32)
            # print(np.min(img_final), np.max(img_final))
            return img_final/255.0

    # --- CutMix Implementations ---
    def _apply_cutmix_v1(self, images, labels_onehot):
        # ... (Copy implementation from your existing PatchDataset) ...
        pass # Placeholder: Insert your existing code here

    def _apply_cutmix_v2(self, images, labels_onehot, probability=1.0):
        # ... (Copy implementation from your existing PatchDataset) ...
        pass # Placeholder: Insert your existing code here

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.df))
        batch_idx = self.indices[start_idx:end_idx]
        
        if len(batch_idx) == 0: return np.array([]), np.array([])

        batch_df = self.df.iloc[batch_idx]

        # Load images (ROI extraction happens inside here)
        # images_loaded = [self._load_and_preprocess_image(row) for _, row in batch_df.iterrows()]
        labels = batch_df['label'].tolist()       
        images_loaded = []

        for _, row in batch_df.iterrows():
            # print(type(row))
            # break
            if self.expect_rgba:
                img = self._load_and_preprocess_image(row)
            else:
                img = self._load_and_preprocess_image_rgb(row)
            images_loaded.append(img)

        # X_tensor = tf.stack(images_loaded)
        # y_tensor = tf.one_hot(labels, depth=2) # Assumes 2 classes

        # # Apply CutMix if enabled
        # if self.use_cutmix and len(batch_idx) >= 2:
        #      # Note: You need to copy your _apply_cutmix_v1/v2 code into this class
        #      if self.cutmix_version == 'v1': X_out, y_out = self._apply_cutmix_v1(X_tensor, y_tensor)
        #      elif self.cutmix_version == 'v2': X_out, y_out = self._apply_cutmix_v2(X_tensor, y_tensor)
        #      else: X_out, y_out = X_tensor, y_tensor
        # else:
        #      X_out, y_out = X_tensor, y_tensor

        # return X_out.numpy(), y_out.numpy()
        
        # ---------------------------------------------------------
        # CRITICAL FIX for Multiprocessing:
        # Use Pure NumPy instead of TensorFlow operations here.
        # Using tf.* inside __getitem__ triggers CUDA initialization
        # in worker processes, causing conflicts with the main process.
        # ---------------------------------------------------------
        
        # Convert list of images to a NumPy array (Float32)
        X_out = np.array(images_loaded, dtype=np.float32)
        
        # Perform One-Hot Encoding using NumPy (equivalent to tf.one_hot)
        # Assumes 2 classes. 'labels' is a list of integers [0, 1, ...]
        y_out = np.eye(2, dtype=np.float32)[labels] 

        # Handle CutMix (If enabled)
        # Note: Ensure _apply_cutmix_v1/v2 are compatible with NumPy arrays
        # or temporarily disable it to verify stability.
        if self.use_cutmix and len(batch_idx) >= 2:
             if self.cutmix_version == 'v1': 
                 X_out, y_out = self._apply_cutmix_v1(X_out, y_out)
             elif self.cutmix_version == 'v2': 
                 X_out, y_out = self._apply_cutmix_v2(X_out, y_out)
        
        return X_out, y_out
    
    def plot_random_batch(self, batch_idx=None,
                              nrows=4, ncols=8,
                              note=None,
                              save_dir=None):
            """
            Plot a batch of images with optional de-processing for visualization.
            ALIGNMENT UPDATE: Now handles BGR->RGB conversion for display if expect_rgba is True.
            """
        
            # -------- Select batch ----------
            if batch_idx is None and self.shuffle is True:
                batch_idx = np.random.randint(0, len(self))
        
            print(f"\n[PLOT] Batch index: {batch_idx}")
        
            images, labels = self.__getitem__(batch_idx)
            hard_labels = np.argmax(labels, axis=-1)
        
            B = images.shape[0]
            num_to_plot = min(nrows * ncols, B)
        
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(ncols * 2.5, nrows * 2.5),
                dpi=300
            )
        
            if nrows * ncols == 1:
                axes = np.array([axes])
            axes = axes.ravel()
        
            # ---------- Helper: deprocess for visualization ----------
            def visualize_image(img):
                """
                Guarantee valid visualization regardless of preprocessing.
                """
                img = img.copy()
                
                # --- COLOR CORRECTION LOGIC (NEW) ---
                # If the model expects RGBA mode, we know the underlying data is BGR.
                # We must swap channels for Matplotlib to display Red correctly.
                if self.expect_rgba:
                    # Assuming img is float [0, 1] or preprocessed float
                    # We normalize to [0, 1] first to be safe
                    vmin, vmax = img.min(), img.max()
                    if vmax > 1.0 or vmin < 0.0:
                         img_norm = (img - vmin) / (vmax - vmin + 1e-7)
                    else:
                         img_norm = img
                    
                    # Convert to uint8 for robust colorspace conversion
                    img_uint8 = (img_norm * 255).astype(np.uint8)
                    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
                    
                    # Convert back to float for consistent plotting
                    img = img_rgb.astype(np.float32) / 255.0
    
                # --- Standard Normalization (Existing) ---
                # If the image looks preprocessed (e.g., ConvNeXt: [0,1]),
                # Normalize to [0,1] for imshow safety.
                vmin, vmax = img.min(), img.max()
                if vmax > 1.0 or vmin < 0.0:
                    img = (img - vmin) / (vmax - vmin + 1e-7)
        
                # Handle grayscale unexpectedly
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
        
                return img.clip(0,1)
        
            # ---------- Plot patches ----------
            for i in range(num_to_plot):
                ax = axes[i]
                img = images[i]
                lab = hard_labels[i]
                soft = labels[i]
        
                # Use the updated helper
                img_vis = visualize_image(img)
        
                ax.imshow(img_vis)
                ax.set_xticks([]); ax.set_yticks([])
        
                # Title text
                if self.use_cutmix:
                    ax.set_title(f"[{soft[0]:.2f}, {soft[1]:.2f}]", fontsize=7)
                else:
                    ax.set_title(f"{lab}", fontsize=20)
        
                # Light frame
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color("lightgray")
                    s.set_linewidth(0.8)
        
            # Hide leftover axes
            for i in range(num_to_plot, len(axes)):
                axes[i].axis("off")
            
            if note is not None:
                if self.preprocessing_function is not None:
                    title = note + ' (Preprocessed Input)'
                    fig.suptitle(title, fontsize=20)
                else:
                    fig.suptitle(note, fontsize=20)
              
            plt.tight_layout()
            plt.show()
        
            # ---------- Save --------
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f"batch_{batch_idx}_viz.png")
                fig.savefig(fname)
                print(f"Saved visualization → {fname}")


#%%