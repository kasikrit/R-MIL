import numpy as np
import tensorflow as tf
from sklearn.covariance import LedoitWolf
from tensorflow.keras import layers, Model

class RiemannianMetrics:
    """
    Implements Riemannian Geometry metrics for Latent Space Analysis.
    Phase 1: Metric Tensor Estimation via Inverse Covariance.
    Phase 2: Scalar Curvature (Mahalanobis Energy) calculation.
    """
    
    def __init__(self, n_classes=2, feature_dim=16):
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.means = {}         # Class Centroids (mu)
        self.precisions = {}    # Inverse Covariance Matrices (Sigma^-1)
        self.threshold = None   # Curvature Threshold

    def fit_manifold(self, features, labels):
        """
        [Phase 1] Estimate the Riemannian Manifold parameters (Centroids & Precision Matrices).
        Using Ledoit-Wolf shrinkage for robust covariance estimation.
        """
        print(f"\n[RiemannianMetrics] Fitting Manifold on {len(features)} samples...")
        
        for c in range(self.n_classes):
            # Extract features for class c
            class_feats = features[labels == c]
            
            if len(class_feats) < self.feature_dim:
                print(f"  Warning: Class {c} has too few samples ({len(class_feats)}) for robust estimation.")
            
            # 1. Calculate Centroid (mu)
            mu = np.mean(class_feats, axis=0)
            self.means[c] = mu
            
            # 2. Calculate Precision Matrix (Sigma^-1)
            # Use LedoitWolf for better stability in high dimensions/low sample size
            cov_estimator = LedoitWolf().fit(class_feats)
            precision = cov_estimator.precision_
            self.precisions[c] = precision
            
            print(f"  Class {c}: Centroid shape {mu.shape}, Precision shape {precision.shape}")

    def compute_curvature_energy(self, z):
        """
        [Phase 1/2] Compute Scalar Curvature (Energy) for a single feature vector z.
        Formula: E(z) = min_c sqrt( (z - mu_c)^T * Sigma_c^-1 * (z - mu_c) )
        """
        energies = []
        for c in range(self.n_classes):
            mu = self.means[c]
            P = self.precisions[c]
            
            diff = z - mu
            # Mahalanobis Distance Squared: d^2 = diff.T * P * diff
            dist_sq = np.dot(np.dot(diff, P), diff.T)
            
            # Energy = Sqrt(Distance Squared) approx Scalar Curvature distance
            energies.append(np.sqrt(dist_sq))
            
        # Return the minimum energy distance to ANY known class manifold
        return min(energies)
    
    
    def compute_batch_curvature(self, batch_features):
        """
        Vectorized curvature computation using NumPy only.
        This function is intentionally NOT TensorFlow-traceable.
        """
    
        # ---- 1. Force eager-only execution (never traced by TF) ----
        # If this function is called inside a tf.function, disable autograph here.
        try:
            # Works only inside graph; ignored otherwise.
            tf.autograph.experimental.do_not_convert()
        except Exception:
            pass
    
        # ---- 2. Force batch_features to NumPy ----
        if isinstance(batch_features, tf.Tensor):
            batch_features = batch_features.numpy()
        elif hasattr(batch_features, "numpy"):
            batch_features = batch_features.numpy()
        else:
            batch_features = np.array(batch_features)
    
        # ---- 3. Python-level iteration (NOT allowed in TensorFlow graphs) ----
        # Now safe because we ensured pure NumPy mode.
        energies = np.array([
            self.compute_curvature_energy(z)
            for z in batch_features
        ])
    
        return energies

    def calibrate_threshold(self, val_features, percentile=90):
        """
        [Phase 2] Calibrate the Curvature Threshold using Validation Data.
        We assume high energy = anomaly/informative. 
        Usually, we want to filter out the 'flat' generic cells (low energy) or extreme outliers.
        Here, we define a 'Generic Threshold'. 
        If Energy < Threshold -> Generic (Noise).
        If Energy >= Threshold -> Informative (Signal).
        """
        energies = self.compute_batch_curvature(val_features)
        self.threshold = np.percentile(energies, percentile)
        print(f"\n[RiemannianMetrics] Calibrated Threshold @ {percentile}th percentile: {self.threshold:.4f}")
        return self.threshold

class CurvatureAttention(tf.keras.layers.Layer):
    """
    [Phase 2] Attention Mechanism Layer based on Curvature.
    Reweights patch features based on their pre-computed curvature scores.
    """
    def __init__(self, **kwargs):
        super(CurvatureAttention, self).__init__(**kwargs)
        
        # Learnable parameters for the sigmoid gate
        # weight = sigmoid( alpha * energy + beta )
        self.alpha = self.add_weight(name='alpha', shape=(1,), initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(1,), initializer='zeros', trainable=True)

    def call(self, inputs):
        # inputs: [features (B, D), energies (B, 1)]
        features, energies = inputs
        
        # Calculate Attention Weights
        weights = tf.math.sigmoid(self.alpha * energies + self.beta)
        
        # Apply weights to features
        weighted_features = features * weights
        
        return weighted_features, weights

import tensorflow as tf
from tensorflow.keras import layers, Model

class PoincareMath:
    """
    Utility class for Differentiable Poincaré Ball operations in TensorFlow.
    """
    
    @staticmethod
    def exp_map(x, c=1.0):
        """Maps Euclidean vector x to the Poincaré ball."""
        # Formula: tanh(sqrt(c)/2 * ||x||) * (x / (sqrt(c) * ||x||))
        x_norm = tf.norm(x, axis=-1, keepdims=True) + 1e-10
        lambda_x = tf.tanh(tf.sqrt(c) * x_norm / 2)
        return lambda_x * (x / (tf.sqrt(c) * x_norm))

    @staticmethod
    def log_map(y, c=1.0):
        """Maps Poincaré point y back to Euclidean tangent space at origin."""
        # Formula: (2/sqrt(c)) * arctanh(sqrt(c) * ||y||) * (y / ||y||)
        y_norm = tf.norm(y, axis=-1, keepdims=True)
        # Clip norm to avoid numerical instability at boundary (norm=1)
        y_norm = tf.clip_by_value(y_norm, 0, 1 - 1e-5)
        
        coef = (2 / tf.sqrt(c)) * tf.math.atanh(tf.sqrt(c) * y_norm)
        return coef * (y / (y_norm + 1e-10))

    @staticmethod
    def mobius_add(x, y, c=1.0):
        """Hyperbolic addition (Möbius addition) of vectors x and y."""
        # Numerator
        xy = tf.reduce_sum(x * y, axis=-1, keepdims=True) # Inner product <x,y>
        x2 = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) # ||x||^2
        y2 = tf.reduce_sum(tf.square(y), axis=-1, keepdims=True) # ||y||^2
        
        num = (1 + 2*c*xy + c*y2)*x + (1 - c*x2)*y
        denom = 1 + 2*c*xy + c**2 * x2 * y2 + 1e-10
        
        return num / denom
    
    @staticmethod
    def mobius_matvec(M, x, c=1.0):
        """
        Matrix-vector multiplication in Hyperbolic space.
        Approximation: Log -> Linear -> Exp (Tangent space operation)
        """
        # 1. Map to Euclidean Tangent space
        x_tan = PoincareMath.log_map(x, c)
        # 2. Apply Linear Transform (Mx)
        mx_tan = tf.matmul(x_tan, M) 
        # 3. Map back to Hyperbolic
        return PoincareMath.exp_map(mx_tan, c)

class HyperbolicDense(layers.Layer):
    """
    A Hyperbolic alternative to the standard Dense layer.
    Performs: y = (M @ x) (+) b in Poincaré space.
    """
    def __init__(self, units, c=1.0, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.c = c
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(
            name='weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # 1. Linear Transform (Möbius MatVec)
        # We assume inputs are already in the Ball (or projected there)
        res = PoincareMath.mobius_matvec(self.w, inputs, self.c)
        
        # 2. Bias Addition (Möbius Add)
        # Reshape bias to broadcast: (1, 1, Units)
        bias_reshaped = tf.reshape(self.b, (1, 1, self.units))
        # Project bias to manifold before adding
        bias_hyp = PoincareMath.exp_map(bias_reshaped, self.c)
        
        output = PoincareMath.mobius_add(res, bias_hyp, self.c)
        
        if self.activation:
            # Note: Applying Euclidean activation on Manifold is theoretically debated.
            # A safer path is Log -> Act -> Exp, but for ReLU we often skip or operate in tangent.
            # Here we act in tangent space for stability.
            tan = PoincareMath.log_map(output, self.c)
            act = self.activation(tan)
            output = PoincareMath.exp_map(act, self.c)
            
        return output

class FrechetMean(layers.Layer):
    """
    Calculates the weighted Fréchet Mean (Geometric Center of Mass).
    Using the 'Einstein Midpoint' closed-form approximation in the Klein model,
    which is isometric to Poincaré.
    """
    def __init__(self, c=1.0, **kwargs):
        super().__init__(**kwargs)
        self.c = c

    def call(self, inputs):
        # inputs[0]: Features (Batch, N, D) [in Poincaré Ball]
        # inputs[1]: Weights (Batch, N, 1) [Normalized Attention Scores]
        x, w = inputs
        
        # 1. Compute Lorentz Factors (Gamma)
        # Gamma = 1 / sqrt(1 - c * ||x||^2)
        x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        # Clip for numerical safety
        x_norm_sq = tf.clip_by_value(x_norm_sq, 0, (1/self.c) - 1e-5)
        gamma = 1.0 / tf.sqrt(1.0 - self.c * x_norm_sq)
        
        # 2. Compute Einstein Weighted Sum (in Klein coordinates)
        # The sum is performed on (Gamma * w * x)
        weighted_gamma = w * gamma
        
        numerator = tf.reduce_sum(weighted_gamma * x, axis=1) # Sum over patches
        denominator = tf.reduce_sum(weighted_gamma, axis=1) - 1e-7 # Sum over patches
        
        # 3. Result (Klein Model point)
        mean_klein = numerator / denominator
        
        # 4. Convert Klein -> Poincaré (if needed for next layers, though usually we output here)
        # Klein to Poincare: p = k / (1 + sqrt(1 - c*||k||^2))
        k_norm_sq = tf.reduce_sum(tf.square(mean_klein), axis=-1, keepdims=True)
        k_norm_sq = tf.clip_by_value(k_norm_sq, 0, (1/self.c) - 1e-5)
        
        mean_poincare = mean_klein / (1.0 + tf.sqrt(1.0 - self.c * k_norm_sq))
        
        return mean_poincare
    

