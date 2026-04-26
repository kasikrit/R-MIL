import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from yacs.config import CfgNode as CN
# import configsAneTunedHP_2 as cf
# config = cf.get_config()

# import configsCell
# config = configsCell.get_config()

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# Assuming 'config' is available and defines necessary constants

class PatchDataset(tf.keras.utils.Sequence):
    def __init__(self, config, df, batch_size=32,
                 preprocessing_function=None,
                 shuffle=True,
                 save=False,
                 use_cutmix=False,
                 cutmix_alpha=1.0,
                 cutmix_version='v1',
                 verbose=False):
        self.config = config
        self.df = df.reset_index(drop=True)
        print(f"{len(df)=}")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.save = save
        self.verbose = verbose
        self.preprocessing_function = preprocessing_function
        print(f"Using preprocessing function: {self.preprocessing_function}") # More informative print

        # --- CutMix Initialization ---
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_version = cutmix_version
        if self.use_cutmix:
            print(f"CutMix augmentation enabled (Version: {self.cutmix_version}) with alpha={self.cutmix_alpha}")
            if cutmix_version not in ['v1', 'v2']:
                raise ValueError("cutmix_version must be 'v1' or 'v2'")

        self.on_epoch_end()

    # --- Properties and Basic Methods ---
    @property
    def classes(self):
        return self.df['label'].to_numpy()

    def __len__(self):
        # Ensure the last partial batch is included
        return int(np.ceil(len(self.df) / self.batch_size))


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)  

    # --- Image Loading and Initial Handling ---
    def _load_and_preprocess_image(self, row):
        """Loads image, handles color, resizes, and calls model preprocessing."""
        img_path = row['patch_path']
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # --- Handle color modes -> Ensure RGB uint8 [0, 255] ---
        if len(img.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:  # BGRA/RGBA -> RGB
            # Assuming input might be BGRA from cv2, convert to RGBA then RGB
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
        elif img.shape[-1] == 3:  # BGR -> RGB
             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape} for path: {img_path}")

        # --- Convert to Float [0, 1] ---
        # This is the standard format before model-specific preprocessing or augmentations
        img_rgb_float = img_rgb.astype(np.float32) / 255.0        
        
        return img_rgb_float # Return Tensor

    # --- CutMix Implementations (Keep _apply_cutmix_v1 and _apply_cutmix_v2) ---
    def _apply_cutmix_v1(self, images, labels_onehot):
        # (Implementation from previous response)
        """Applies CutMix augmentation (Beta Dist Box) to a batch."""
        batch_size = tf.shape(images)[0]
        img_h, img_w = tf.shape(images)[1], tf.shape(images)[2]

        indices = tf.random.shuffle(tf.range(batch_size))
        images_shuffled = tf.gather(images, indices)
        labels_shuffled = tf.gather(labels_onehot, indices)

        lam = tf.compat.v1.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
        
        img_h_float = tf.cast(img_h, tf.float32)
        img_w_float = tf.cast(img_w, tf.float32)

        cx = tf.random.uniform(shape=[], minval=0, maxval=img_w, dtype=tf.int32)
        cy = tf.random.uniform(shape=[], minval=0, maxval=img_h, dtype=tf.int32)
        ratio = tf.math.sqrt(1.0 - lam)
        h = tf.cast(img_h_float * ratio, tf.int32)
        w = tf.cast(img_w_float * ratio, tf.int32)

        y1 = tf.clip_by_value(cy - h // 2, 0, img_h)
        y2 = tf.clip_by_value(cy + h // 2, 0, img_h)
        x1 = tf.clip_by_value(cx - w // 2, 0, img_w)
        x2 = tf.clip_by_value(cx + w // 2, 0, img_w)

        patch = images_shuffled[:, y1:y2, x1:x2, :]

        mask_updates = tf.ones_like(patch, dtype=tf.float32)
        paddings = [[0, 0], [y1, img_h - y2], [x1, img_w - x2], [0, 0]]
        mask_tensor = tf.pad(mask_updates, paddings, "CONSTANT", constant_values=0.0)
        patch_padded = tf.pad(patch, paddings, "CONSTANT", constant_values=0.0)

        images_cutmix = images * (1.0 - mask_tensor) + patch_padded

        effective_lam = 1.0 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / (img_w_float * img_h_float)
        labels_cutmix = effective_lam * labels_onehot + (1.0 - effective_lam) * labels_shuffled

        return images_cutmix, labels_cutmix


    def _apply_cutmix_v2(self, images, labels_onehot, idx=None, probability=1.0):
        """
        CutMix V2: one random rectangular patch per image, always attempting cross-class mixing.
    
        Args:
            images: Tensor (B, H, W, C)
            labels_onehot: Tensor (B, num_classes)
            idx: batch index (optional, for logging)
            probability: probability of applying CutMix per image
        Returns:
            images_cutmix, labels_cutmix
        """
        batch_size = tf.shape(images)[0]
        img_h, img_w = tf.shape(images)[1], tf.shape(images)[2]
        num_classes = tf.shape(labels_onehot)[1]
    
        imgs, labs = [], []
    
        for j in tf.range(batch_size):
            # Probability gate
            apply_cutmix = tf.random.uniform([], 0.0, 1.0) <= probability
    
            # --- Partner selection (prefer different class) ---
            possible_k = tf.range(batch_size)
            possible_k = tf.boolean_mask(possible_k, possible_k != j)
            k = tf.random.shuffle(possible_k)[0] if tf.size(possible_k) > 0 else j
            
            img_j = images[j]  # anchor image
            img_k = images[k] 
   
            label_j = labels_onehot[j]
            label_candidates = tf.gather(labels_onehot, possible_k)
            # Find candidates with different class
            diff_mask = tf.reduce_any(tf.not_equal(label_candidates, label_j), axis=1)
            diff_class_idx = tf.boolean_mask(possible_k, diff_mask)
    
            # Use different class if available, else fallback to any
            candidate_pool = tf.cond(tf.size(diff_class_idx) > 0,
                                     lambda: diff_class_idx,
                                     lambda: possible_k)
            k_idx = tf.random.uniform([], 0, tf.cast(tf.shape(candidate_pool)[0], tf.float32))
            k_idx = tf.cast(k_idx, tf.int32)
            k = tf.gather(candidate_pool, tf.clip_by_value(k_idx, 0, tf.shape(candidate_pool)[0] - 1))
    
            # --- CutMix box definition ---
            lam = tf.compat.v1.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
            lam = tf.clip_by_value(lam, 0.2, 0.8)
    
            cut_w = tf.cast(tf.cast(img_w, tf.float32) * tf.sqrt(1.0 - lam), tf.int32)
            cut_h = tf.cast(tf.cast(img_h, tf.float32) * tf.sqrt(1.0 - lam), tf.int32)
    
            cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
            cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
    
            x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
            x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
            y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
            y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)
    
            # --- Compose mixed image safely ---
            # Get coordinate grid (y,x) for patch area
            ys, xs = tf.meshgrid(tf.range(y1, y2), tf.range(x1, x2), indexing="ij")
            coords = tf.stack([ys, xs], axis=-1)            # shape (h, w, 2)
            coords_flat = tf.reshape(coords, [-1, 2])       # shape (h*w, 2)
            
            # Flatten patch from img_k
            patch_flat = tf.reshape(img_k[y1:y2, x1:x2, :], [-1, tf.shape(images)[-1]])  # (h*w, c)
            
            # Apply update (replaces pixels in img_j)
            mixed_img = tf.tensor_scatter_nd_update(img_j, coords_flat, patch_flat)

    
            # --- Effective lambda (actual area ratio) ---
            area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
            total = tf.cast(img_w * img_h, tf.float32)
            effective_lam = 1.0 - (area / tf.maximum(total, 1e-6))
    
            # --- Mix labels ---
            lab1 = tf.cast(label_j, tf.float32)
            lab2 = tf.cast(labels_onehot[k], tf.float32)
            mixed_lab = effective_lam * lab1 + (1.0 - effective_lam) * lab2
    
            imgs.append(tf.cond(apply_cutmix, lambda: mixed_img, lambda: img_j))
            labs.append(tf.cond(apply_cutmix, lambda: mixed_lab, lambda: lab1))
    
            # --- Debug print for first few only ---
            if self.verbose:
                if j < 3:
                    tf.print("\n--- CutMix Debug Sample", j, "---")
                    tf.print("Partner k:", k)
                    tf.print("Lam (Beta):", lam)
                    tf.print("Effective Lam:", effective_lam)
                    tf.print("Labels:", lab1, "+", lab2, "=>", mixed_lab)
                    tf.print("Box:", x1, y1, x2, y2)
    
        images_cutmix = tf.stack(imgs)
        labels_cutmix = tf.stack(labs)
        images_cutmix = tf.reshape(images_cutmix, (batch_size, img_h, img_w, tf.shape(images)[-1]))
        labels_cutmix = tf.reshape(labels_cutmix, (batch_size, num_classes))
    
        return images_cutmix, labels_cutmix
    
    def cutmix(self, image, label, PROBABILITY = 1.0):
        # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
        # output - a batch of images with cutmix applied
        AUG_BATCH = self.batch_size
        DIM = self.config.DATA.IMAGE_SIZE #256
        CLASSES = self.config.DATA.CLASSES
        
        imgs = []; labs = []
        for j in range(AUG_BATCH):
            # AUG_BATCH - size for the augmentation batch.
            # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
            
            P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.int32)
            
            # CHOOSE RANDOM IMAGE TO CUTMIX WITH
            k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)
            
            # CHOOSE RANDOM LOCATION
            x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
            y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
            b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0
            WIDTH = tf.cast( DIM * tf.math.sqrt(1-b),tf.int32) * P
            ya = tf.math.maximum(0,y-WIDTH//2)
            yb = tf.math.minimum(DIM,y+WIDTH//2)
            xa = tf.math.maximum(0,x-WIDTH//2)
            xb = tf.math.minimum(DIM,x+WIDTH//2)
            
            # MAKE CUTMIX IMAGE
            one = image[j,ya:yb,0:xa,:]
            two = image[k,ya:yb,xa:xb,:]
            three = image[j,ya:yb,xb:DIM,:]
            middle = tf.concat([one,two,three],axis=1)
            img = tf.concat([image[j,0:ya,:,:],middle 
                                 ,image[j,yb:DIM,:,:]],axis=0)
            imgs.append(img)
            
            # MAKE CUTMIX LABEL
            a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)
            if len(label.shape)==1:
                lab1 = tf.one_hot(label[j],CLASSES)
                lab2 = tf.one_hot(label[k],CLASSES)
            else:
                lab1 = label[j,]
                lab2 = label[k,]
            labs.append((1-a)*lab1 + a*lab2)
                
        image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))
        label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))
        return image2,label2

    
    def _apply_cutmix_v2_force_diff_class(self, images, labels_onehot, probability=1.0):
        """
        CutMix v3: Guaranteed fractional labels if the batch contains ≥2 classes.
        Ensures correct geometric placement of patches (no black padding artifacts).
        """
        batch_size = tf.shape(images)[0]
        img_h, img_w = tf.shape(images)[1], tf.shape(images)[2]

        # Convert one-hot to class indices
        labels_class = tf.argmax(labels_onehot, axis=1)

        imgs, labs = [], []

        for j in tf.range(batch_size):
            apply_cutmix = tf.random.uniform([], 0.0, 1.0) <= probability
            label_j = labels_onehot[j]
            class_j = labels_class[j]

            # --- Find indices with different class ---
            diff_idx = tf.where(tf.not_equal(labels_class, class_j))[:, 0]
            no_diff = tf.equal(tf.size(diff_idx), 0)

            # Choose partner from different class (or itself if all same)
            k = tf.cond(
                no_diff,
                lambda: j,
                lambda: diff_idx[tf.random.uniform([], 0, tf.shape(diff_idx)[0], dtype=tf.int32)]
            )

            img_j = images[j]
            img_k = images[k]
            label_k = labels_onehot[k]

            # --- Sample lambda ---
            lam = tf.compat.v1.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
            lam = tf.clip_by_value(lam, 0.2, 0.8)

            # --- Compute patch coordinates ---
            cut_w = tf.cast(tf.cast(img_w, tf.float32) * tf.sqrt(1.0 - lam), tf.int32)
            cut_h = tf.cast(tf.cast(img_h, tf.float32) * tf.sqrt(1.0 - lam), tf.int32)
            cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
            cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)
            x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
            x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
            y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
            y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)

            # ==========================================================
            # --- Correct mixed image construction (no resize, no black) ---
            # Start from base image
            mixed_img = tf.identity(img_j)

            # Extract donor patch (exact region)
            patch = img_k[y1:y2, x1:x2, :]

            # Create coordinate grid for patch insertion
            yy, xx = tf.meshgrid(tf.range(y1, y2), tf.range(x1, x2), indexing="ij")
            coords = tf.stack([yy, xx], axis=-1)
            coords_flat = tf.reshape(coords, [-1, 2])
            patch_flat = tf.reshape(patch, [-1, tf.shape(img_j)[-1]])

            # Create full-size mask and full-size padded patch
            mask = tf.tensor_scatter_nd_update(
                tf.zeros_like(img_j),
                coords_flat,
                tf.ones_like(patch_flat)
            )
            padded_patch = tf.tensor_scatter_nd_update(
                tf.zeros_like(img_j),
                coords_flat,
                patch_flat
            )

            # Blend patch and base
            mixed_img = img_j * (1.0 - mask) + padded_patch
            # ==========================================================

            # --- Compute label mix ---
            area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
            total = tf.cast(img_w * img_h, tf.float32)
            effective_lam = 1.0 - (area / tf.maximum(total, 1e-6))
            mixed_lab = effective_lam * label_j + (1.0 - effective_lam) * label_k

            # If only one class in batch → skip mix
            final_img = tf.cond(apply_cutmix & ~no_diff, lambda: mixed_img, lambda: img_j)
            final_lab = tf.cond(apply_cutmix & ~no_diff, lambda: mixed_lab, lambda: label_j)

            imgs.append(final_img)
            labs.append(final_lab)

            # --- Debug prints ---
            if self.verbose:
                if j < 3:
                    tf.print("\n--- CutMix Sample", j, "---")
                    tf.print("Class_j:", class_j, "Partner_k:", k, "Class_k:", labels_class[k])
                    tf.print("Box:", x1, y1, x2, y2, "Effective Lam:", effective_lam)
                    tf.print("Label_j:", label_j, "Label_k:", label_k, "Mixed label:", mixed_lab)
                    tf.print("ApplyCutMix:", apply_cutmix, "NoDiff:", no_diff)

        return tf.stack(imgs), tf.stack(labs)


    # --- Get Batch ---
    def __getitem__(self, idx):
        # start_idx = idx * self.batch_size
        # end_idx = min(start_idx + self.batch_size, len(self.df))
        # batch_idx = self.indices[start_idx:end_idx]
        
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.df))
        batch_df = self.df.iloc[start:end]
        
        actual_batch_size = len(batch_df)

        # Load images (basic handling: color, float[0,1])
        # _load_basic_image should return float32 numpy RGB [0,1]
        images_loaded = [self._load_basic_image(row) for _, row in batch_df.iterrows()]
        labels = batch_df['label'].tolist()

        # Stack images into a batch Tensor
        X_tensor = tf.stack(images_loaded)
        # Convert labels to one-hot encoding Tensor
        y_tensor = tf.one_hot(labels, depth=self.config.MODEL.NUM_CLASSES)

        # --- Apply Selected CutMix Version ---
        if self.use_cutmix and actual_batch_size >= 2:
            if self.cutmix_version == 'v1':
                X_aug, y_aug = self._apply_cutmix_v1(X_tensor, y_tensor)
            elif self.cutmix_version == 'v2':
                # X_aug, y_aug = self._apply_cutmix_v2(
                #     X_tensor,
                #     y_tensor,
                #     idx,
                #     probability=1.0)
                
                X_aug, y_aug = self._apply_cutmix_v2_force_diff_class(
                    X_tensor,
                    y_tensor,
                    probability=1.0)
                
                # X_aug, y_aug = self.cutmix(X_tensor, y_tensor)
            else:
                 X_aug, y_aug = X_tensor, y_tensor # Fallback
        else:
             X_aug, y_aug = X_tensor, y_tensor # No CutMix

        # --- Apply Model Preprocessing Individually ---
        processed_images = []
        # Iterate through images in the batch (which might be augmented)
        for i in range(actual_batch_size):
            img_tensor = X_aug[i] # Get single image tensor (H, W, C)
            img_processed = self._apply_model_preprocessing(img_tensor) # Apply preprocessing
            # Ensure shape after preprocessing is correct for a single image
            img_processed = tf.ensure_shape(img_processed, [self.config.DATA.H, self.config.DATA.W, 3])
            processed_images.append(img_processed)

        # Stack the individually processed images back into a batch
        X_final_processed = tf.stack(processed_images)

        return X_final_processed, y_aug  # Return as Tensor, not NumPy

    # --- Renamed Image Loading ---
    # Renamed _load_and_preprocess_image to _load_basic_image
    def _load_basic_image(self, row):
        """Loads image, handles color, ensures RGB float32 [0,1]. NO model preprocessing."""
        img_path = row['patch_path']
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # --- Handle color modes -> Ensure RGB uint8 [0, 255] ---
        if len(img.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:  # BGRA/RGBA -> RGB
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
        elif img.shape[-1] == 3:  # BGR -> RGB
             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape} for path: {img_path}")

        # --- Convert to Float [0, 1] ---
        img_rgb_float = img_rgb.astype(np.float32) / 255.0

        return img_rgb_float # Return Numpy array [0,1] range

    def _apply_model_preprocessing(self, img_rgb_float_tensor):
         """Applies the model-specific preprocessing function or basic normalization."""
         if self.preprocessing_function is not None:
             img_rgb_float_np = img_rgb_float_tensor.numpy()
             image_uint8_bgr = cv2.cvtColor((img_rgb_float_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
             img_preprocessed = self.preprocessing_function(image_uint8_bgr)
             img_final = tf.cast(img_preprocessed, tf.float32)
         else:
             img_final = tf.cast(img_rgb_float_tensor, tf.float32)
         return img_final

    def summary(self):
        # (Implementation from previous response)
        """Quick inspection of dataset composition."""
        total_patches = len(self.df)
        label_counts = self.df['label'].value_counts().to_dict()

        print(f"Dataset size: {total_patches} patches")
        print("Patch counts per label:")
        for label, count in label_counts.items():
            pct = (count / total_patches) * 100
            print(f"  Label {label}: {count} patches ({pct:.2f}%)")

        summary_dict = {
            'total_patches': total_patches,
            'label_counts': label_counts
        }
        return summary_dict

    def plot_random_batch(self, nrows=5, ncols=5):
        # (Implementation from previous response)
        rand_idx = np.random.randint(0, len(self))
        images, labels = self.__getitem__(rand_idx)
        hard_labels = np.argmax(labels, axis=-1)

        num_to_plot = min(nrows * ncols, images.shape[0])
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=300, figsize=(ncols*2, nrows*2))
        if nrows * ncols == 1: axes = np.array([axes])
        axes = axes.ravel()

        for i in range(num_to_plot):
            ax = axes[i]
            img = images[i].numpy()
            label = hard_labels[i]
            soft_label = labels[i]

            if img.min() < -1e-5 or img.max() > 1.0 + 1e-5:
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)

            # Check if image needs channel conversion for display (e.g., BGR from preproc?)
            # Assuming final output is displayable as RGB/Grayscale
            ax.imshow(img)
            if self.use_cutmix:
                # Format soft label array to 2 decimal places for display
                formatted_soft_label = f"[{soft_label[0]:.2f}, {soft_label[1]:.2f}]"
                ax.set_title(formatted_soft_label, fontsize=14) # Use smaller font for array
            else:
                ax.set_title(label, fontsize=14) # Keep hard label for non-cutmix
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color('lightgray'); spine.set_linewidth(0.8)

        for i in range(num_to_plot, len(axes)): axes[i].axis('off')
        plt.tight_layout()
        plt.show()

        if self.save:
            info_string = f"patch-sanity-check-{rand_idx}.png"
            fig.savefig(f"{info_string}")
            print(f"Saved: {info_string}")


    def metadata(self):
        # (Implementation from previous response)
        info = {
            'total_patches': len(self.df),
            'unique_patients': self.df['patient_id'].nunique(),
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'preprocessing_function': str(self.preprocessing_function),
            'label_distribution': self.df['label'].value_counts().to_dict(),
            'use_cutmix': self.use_cutmix,
            'cutmix_version': self.cutmix_version if self.use_cutmix else None,
            'cutmix_alpha': self.cutmix_alpha if self.use_cutmix else None
        }
        return info
    