"""
LE-JEPA Complete Debugging & Visualization Suite

Comprehensive diagnostics for understanding:
1. Representation collapse detection
2. Training dynamics and instabilities
3. Gradient flow and optimization health
4. Embedding quality and distribution
5. Component-wise analysis

Every metric you need to debug collapse, divergence, and poor training.

Author: Debugging suite
Date: 2026-01-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ==========================================================================
# Comprehensive Metrics Logger
# ==========================================================================
class MetricsLogger:
    """
    Logs all training metrics to disk for later analysis.
    Tracks step-by-step evolution of every important quantity.
    """
    def __init__(self, log_dir="./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.metrics = defaultdict(list)
        self.step = 0
    
    def log(self, metrics_dict):
        """Log a dictionary of metrics"""
        self.metrics['step'].append(self.step)
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
        self.step += 1
    
    def save(self):
        """Save metrics to JSON"""
        save_path = self.log_dir / "metrics.json"
        with open(save_path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
        print(f"Saved metrics to {save_path}")
    
    def plot_metrics(self, save_path=None):
        """Plot all logged metrics"""
        if not self.metrics:
            print("No metrics to plot")
            return
        
        df = pd.DataFrame(self.metrics)
        
        # Determine number of subplots needed
        metric_cols = [c for c in df.columns if c != 'step']
        n_metrics = len(metric_cols)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, metric in enumerate(metric_cols):
            axes[idx].plot(df['step'], df[metric], linewidth=2)
            axes[idx].set_title(metric, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Step')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylabel(metric)
        
        # Hide unused subplots
        for idx in range(len(metric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved metrics plot to {save_path}")
        else:
            plt.savefig(self.log_dir / "metrics_over_time.png", dpi=150, bbox_inches='tight')
        
        plt.close()


# ==========================================================================
# Collapse Detector
# ==========================================================================
class CollapseDetector:
    """
    Detects various forms of representation collapse:
    1. Dimensional collapse (low effective rank)
    2. Clustering collapse (all samples same)
    3. Gradient collapse (vanishing gradients)
    4. Output collapse (constant predictions)
    """
    
    @staticmethod
    def compute_effective_rank(embeddings):
        """
        Effective rank via singular values.
        Lower = more collapsed.
        
        Args:
            embeddings: [N, D] tensor
        Returns:
            effective_rank: scalar (max is D)
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        # Center the data
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # SVD
        try:
            _, S, _ = torch.svd(embeddings.float())
            
            # Normalize singular values
            S = S / S.sum()
            
            # Effective rank: exp(entropy of singular value distribution)
            entropy = -(S * torch.log(S + 1e-12)).sum()
            effective_rank = torch.exp(entropy)
            
            return effective_rank.item()
        except:
            return 0.0
    
    @staticmethod
    def compute_covariance_condition_number(embeddings):
        """
        Condition number of covariance matrix.
        High = some dimensions dominate (potential collapse).
        
        Args:
            embeddings: [N, D] tensor
        Returns:
            condition_number: scalar
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        # Center
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Covariance
        cov = (embeddings.T @ embeddings) / (embeddings.shape[0] - 1)
        
        # Eigenvalues
        try:
            eigenvalues = torch.linalg.eigvalsh(cov.float())
            eigenvalues = eigenvalues[eigenvalues > 1e-12]  # filter near-zero
            
            if len(eigenvalues) == 0:
                return float('inf')
            
            condition_number = eigenvalues.max() / eigenvalues.min()
            return condition_number.item()
        except:
            return float('inf')
    
    @staticmethod
    def compute_cosine_similarity_stats(embeddings):
        """
        Statistics of pairwise cosine similarities.
        High mean similarity = collapse.
        
        Args:
            embeddings: [N, D] tensor
        Returns:
            dict with mean, std, min, max cosine similarity
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        # Subsample if too large
        if embeddings.shape[0] > 500:
            idx = torch.randperm(embeddings.shape[0])[:500]
            embeddings = embeddings[idx]
        
        # Normalize
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Cosine similarity matrix
        cos_sim = embeddings_norm @ embeddings_norm.T
        
        # Remove diagonal
        mask = ~torch.eye(cos_sim.shape[0], dtype=bool, device=cos_sim.device)
        cos_sim_vals = cos_sim[mask]
        
        return {
            'cos_sim_mean': cos_sim_vals.mean().item(),
            'cos_sim_std': cos_sim_vals.std().item(),
            'cos_sim_min': cos_sim_vals.min().item(),
            'cos_sim_max': cos_sim_vals.max().item(),
        }
    
    @staticmethod
    def compute_std_per_dimension(embeddings):
        """
        Standard deviation per dimension.
        Low std = dimension not being used (collapse).
        
        Args:
            embeddings: [N, D] tensor
        Returns:
            dict with mean, min std across dimensions
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        std_per_dim = embeddings.std(dim=0)
        
        return {
            'std_mean': std_per_dim.mean().item(),
            'std_min': std_per_dim.min().item(),
            'std_max': std_per_dim.max().item(),
            'std_median': std_per_dim.median().item(),
        }
    
    @staticmethod
    def detect_collapse(embeddings, threshold_rank_ratio=0.5):
        """
        Main collapse detection function.
        
        Args:
            embeddings: [N, D] or [B, L, D] tensor
            threshold_rank_ratio: if effective_rank/D < this, flag collapse
        Returns:
            dict with collapse indicators and metrics
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        D = embeddings.shape[1]
        
        # Compute all metrics
        eff_rank = CollapseDetector.compute_effective_rank(embeddings)
        cond_num = CollapseDetector.compute_covariance_condition_number(embeddings)
        cos_stats = CollapseDetector.compute_cosine_similarity_stats(embeddings)
        std_stats = CollapseDetector.compute_std_per_dimension(embeddings)
        
        # Collapse indicators
        rank_ratio = eff_rank / D
        is_collapsed_rank = rank_ratio < threshold_rank_ratio
        is_collapsed_similarity = cos_stats['cos_sim_mean'] > 0.99
        is_collapsed_std = std_stats['std_mean'] < 0.01
        
        collapse_detected = is_collapsed_rank or is_collapsed_similarity or is_collapsed_std
        
        return {
            'collapse_detected': collapse_detected,
            'effective_rank': eff_rank,
            'rank_ratio': rank_ratio,
            'condition_number': cond_num,
            **cos_stats,
            **std_stats,
        }


# ==========================================================================
# Gradient Analyzer
# ==========================================================================
class GradientAnalyzer:
    """
    Analyzes gradient flow and health.
    Detects vanishing/exploding gradients.
    """
    
    @staticmethod
    def compute_gradient_norm(model, norm_type=2):
        """Compute global gradient norm"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm
    
    @staticmethod
    def compute_gradient_stats_per_layer(model):
        """
        Compute gradient statistics for each layer.
        Helps identify where gradients vanish/explode.
        """
        stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.abs().max().item(),
                    'min': grad.abs().min().item(),
                    'norm': grad.norm().item(),
                }
        
        return stats
    
    @staticmethod
    def compute_weight_gradient_ratio(model):
        """
        Ratio of gradient norm to weight norm per layer.
        Indicates update magnitude relative to current weights.
        """
        ratios = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                weight_norm = param.data.norm().item()
                grad_norm = param.grad.data.norm().item()
                
                if weight_norm > 1e-12:
                    ratios[name] = grad_norm / weight_norm
                else:
                    ratios[name] = 0.0
        
        return ratios
    
    @staticmethod
    def plot_gradient_flow(model, save_path=None):
        """
        Visualize gradient flow through the network.
        Each bar = average gradient magnitude per layer.
        """
        ave_grads = []
        max_grads = []
        layers = []
        
        for name, param in model.named_parameters():
            if param.grad is not None and "bias" not in name:
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().cpu().item())
                max_grads.append(param.grad.abs().max().cpu().item())
        
        plt.figure(figsize=(14, 6))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", label="max")
        plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b", label="mean")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90, fontsize=8)
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001)
        plt.xlabel("Layers", fontsize=12)
        plt.ylabel("Gradient Magnitude", fontsize=12)
        plt.title("Gradient Flow Through Network", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()


# ==========================================================================
# Embedding Visualizer
# ==========================================================================
class EmbeddingVisualizer:
    """
    Visualizes learned embeddings in various ways.
    """
    
    @staticmethod
    def plot_cosine_similarity_matrix(embeddings, save_path=None, max_samples=100):
        """
        Plot pairwise cosine similarity heatmap.
        Uniform color = collapse.
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        # Subsample
        if embeddings.shape[0] > max_samples:
            idx = torch.randperm(embeddings.shape[0])[:max_samples]
            embeddings = embeddings[idx]
        
        # Normalize
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Cosine similarity
        cos_sim = (embeddings_norm @ embeddings_norm.T).cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cos_sim, cmap='viridis', square=True, 
                    cbar_kws={'label': 'Cosine Similarity'})
        plt.title('Cosine Similarity Matrix\n(Uniform = Collapse)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index')
        plt.ylabel('Sample Index')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    @staticmethod
    def plot_eigenvalue_spectrum(embeddings, save_path=None, min_rank=10):
        """
        Plot eigenvalue spectrum of covariance matrix.
        Robust to low-rank / ill-conditioned matrices.
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))

        # Center
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)

        N, D = embeddings.shape
        if N < D + 1:
            print("⚠️ Too few samples for covariance — skipping eigen spectrum")
            return

        # Covariance
        cov = (embeddings.T @ embeddings) / (N - 1)
        cov = cov.float()

        # Rank check (critical)
        try:
            rank = torch.linalg.matrix_rank(cov).item()
            if rank < min_rank:
                print(f"⚠️ Covariance rank too low ({rank}) — skipping eigen spectrum")
                return
        except RuntimeError:
            print("⚠️ Rank computation failed — skipping eigen spectrum")
            return

        # Diagonal jitter for numerical stability
        eps = 1e-5 * cov.diag().mean()
        cov = cov + eps * torch.eye(D, device=cov.device)

        # Prefer SVD (more stable than eigvalsh)
        try:
            singular_vals = torch.linalg.svdvals(cov)
            eigenvalues = singular_vals.cpu().numpy()
        except RuntimeError:
            print("⚠️ SVD failed — skipping eigen spectrum")
            return

        # Clean & sort
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if len(eigenvalues) == 0:
            print("⚠️ No valid eigenvalues — skipping plot")
            return

        eigenvalues = np.sort(eigenvalues)[::-1]

        # Plot
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(eigenvalues, 'o-', linewidth=2, markersize=4)
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalue Spectrum (Linear)')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.semilogy(eigenvalues, 'o-', linewidth=2, markersize=4)
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalue (log)')
        plt.title('Eigenvalue Spectrum (Log)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close()

    
    @staticmethod
    def plot_pca_projection(embeddings, labels=None, save_path=None, max_samples=500):
        """
        2D PCA projection of embeddings.
        No structure = potential collapse.
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        embeddings_np = embeddings.cpu().numpy()
        
        # Subsample
        if embeddings_np.shape[0] > max_samples:
            idx = np.random.permutation(embeddings_np.shape[0])[:max_samples]
            embeddings_np = embeddings_np[idx]
            if labels is not None:
                labels = labels[idx]
        
        # PCA
        pca = PCA(n_components=2)
        projected = pca.fit_transform(embeddings_np)
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            scatter = plt.scatter(projected[:, 0], projected[:, 1], 
                                c=labels, cmap='tab10', alpha=0.6, s=30)
            plt.colorbar(scatter, label='Label')
        else:
            plt.scatter(projected[:, 0], projected[:, 1], 
                       alpha=0.6, s=30, c='blue')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)', fontsize=12)
        plt.title('PCA Projection of Embeddings\n(No structure = collapse)', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        
        return pca.explained_variance_ratio_
    
    @staticmethod
    def plot_tsne_projection(embeddings, labels=None, save_path=None, max_samples=500):
        """
        2D t-SNE projection of embeddings.
        More sensitive to local structure than PCA.
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        embeddings_np = embeddings.cpu().numpy()
        
        # Subsample
        if embeddings_np.shape[0] > max_samples:
            idx = np.random.permutation(embeddings_np.shape[0])[:max_samples]
            embeddings_np = embeddings_np[idx]
            if labels is not None:
                labels = labels[idx]
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        projected = tsne.fit_transform(embeddings_np)
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            scatter = plt.scatter(projected[:, 0], projected[:, 1], 
                                c=labels, cmap='tab10', alpha=0.6, s=30)
            plt.colorbar(scatter, label='Label')
        else:
            plt.scatter(projected[:, 0], projected[:, 1], 
                       alpha=0.6, s=30, c='blue')
        
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.title('t-SNE Projection of Embeddings\n(Blob = collapse)', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    @staticmethod
    def plot_dimension_usage(embeddings, save_path=None):
        """
        Plot standard deviation per dimension.
        Unused dimensions (low std) indicate collapse.
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        std_per_dim = embeddings.std(dim=0).cpu().numpy()
        mean_per_dim = embeddings.mean(dim=0).cpu().numpy()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Standard deviation
        axes[0].bar(range(len(std_per_dim)), std_per_dim, alpha=0.7)
        axes[0].axhline(y=std_per_dim.mean(), color='r', linestyle='--', 
                       label=f'Mean: {std_per_dim.mean():.4f}')
        axes[0].set_xlabel('Dimension', fontsize=12)
        axes[0].set_ylabel('Standard Deviation', fontsize=12)
        axes[0].set_title('Dimension Usage (Std Dev)\n(Low values = unused dimensions)', 
                         fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Mean
        axes[1].bar(range(len(mean_per_dim)), mean_per_dim, alpha=0.7, color='orange')
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].set_xlabel('Dimension', fontsize=12)
        axes[1].set_ylabel('Mean', fontsize=12)
        axes[1].set_title('Mean per Dimension\n(Should be near 0 for isotropic)', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    @staticmethod
    def plot_correlation_matrix(embeddings, save_path=None, max_dims=50):
        """
        Plot correlation between dimensions.
        High off-diagonal = redundant dimensions.
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        # Limit dimensions for visualization
        if embeddings.shape[1] > max_dims:
            embeddings = embeddings[:, :max_dims]
        
        # Correlation matrix
        embeddings_np = embeddings.cpu().numpy()
        corr = np.corrcoef(embeddings_np.T)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='coolwarm', center=0, 
                    vmin=-1, vmax=1, square=True,
                    cbar_kws={'label': 'Correlation'})
        plt.title('Dimension Correlation Matrix\n(Off-diagonal should be near 0)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Dimension')
        plt.ylabel('Dimension')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()


# ==========================================================================
# Training Monitor (combines all diagnostics)
# ==========================================================================
class TrainingMonitor:
    """
    Comprehensive training monitor that runs all diagnostics
    and saves results at regular intervals.
    """
    
    def __init__(self, log_dir="./debug_logs", check_interval=100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.check_interval = check_interval
        self.step = 0
        
        # Create subdirectories
        (self.log_dir / "embeddings").mkdir(exist_ok=True)
        (self.log_dir / "gradients").mkdir(exist_ok=True)
        (self.log_dir / "collapse").mkdir(exist_ok=True)
        
        self.metrics_logger = MetricsLogger(self.log_dir)
        
        print(f"Training monitor initialized. Logs at: {self.log_dir}")
    
    def check_training_health(self, model, predictions, targets, loss_dict, 
                            batch_idx, epoch):
        """
        Main health check function. Call this during training.
        
        Args:
            model: the model being trained
            predictions: predicted embeddings
            targets: target embeddings
            loss_dict: dict with loss components
            batch_idx: current batch index
            epoch: current epoch
        """
        self.step += 1
        
        # Always log basic metrics
        metrics = {
            'epoch': epoch,
            'batch': batch_idx,
            **loss_dict
        }
        
        # Gradient stats (every step)
        grad_norm = GradientAnalyzer.compute_gradient_norm(model)
        metrics['grad_norm'] = grad_norm
        
        # Collapse detection (every step, but on subset)
        with torch.no_grad():
            collapse_info = CollapseDetector.detect_collapse(predictions)
            metrics.update(collapse_info)
        
        # Log all metrics
        self.metrics_logger.log(metrics)
        
        # Detailed diagnostics at intervals
        if self.step % self.check_interval == 0:
            print(f"\n{'='*60}")
            print(f"DIAGNOSTIC CHECKPOINT - Step {self.step}")
            print(f"{'='*60}")
            
            self._detailed_diagnostics(model, predictions, targets, epoch)
            
            # Save metrics so far
            self.metrics_logger.save()
            self.metrics_logger.plot_metrics(
                self.log_dir / f"metrics_step_{self.step}.png"
            )
            
            print(f"{'='*60}\n")
    
    def _detailed_diagnostics(self, model, predictions, targets, epoch):
        """Run all detailed diagnostics"""
        step_dir = self.log_dir / f"step_{self.step}"
        step_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            # 1. Collapse metrics
            print("\n1. COLLAPSE DETECTION:")
            collapse_pred = CollapseDetector.detect_collapse(predictions)
            collapse_target = CollapseDetector.detect_collapse(targets)
            
            print(f"  Predictions:")
            print(f"    Collapse detected: {collapse_pred['collapse_detected']}")
            print(f"    Effective rank: {collapse_pred['effective_rank']:.2f}")
            print(f"    Rank ratio: {collapse_pred['rank_ratio']:.3f}")
            print(f"    Condition number: {collapse_pred['condition_number']:.2e}")
            print(f"    Cosine sim (mean): {collapse_pred['cos_sim_mean']:.4f}")
            print(f"    Std (mean): {collapse_pred['std_mean']:.4f}")
            
            print(f"  Targets:")
            print(f"    Effective rank: {collapse_target['effective_rank']:.2f}")
            print(f"    Rank ratio: {collapse_target['rank_ratio']:.3f}")
            
            # 2. Gradient analysis
            print("\n2. GRADIENT ANALYSIS:")
            grad_stats = GradientAnalyzer.compute_gradient_stats_per_layer(model)
            grad_ratios = GradientAnalyzer.compute_weight_gradient_ratio(model)
            
            print(f"  Global gradient norm: {GradientAnalyzer.compute_gradient_norm(model):.4e}")
            print(f"  Layers with largest gradients:")
            sorted_grads = sorted(grad_stats.items(), 
                                 key=lambda x: x[1]['norm'], reverse=True)[:5]
            for name, stats in sorted_grads:
                print(f"    {name}: norm={stats['norm']:.4e}, max={stats['max']:.4e}")
            
            print(f"  Layers with smallest gradients:")
            sorted_grads_min = sorted(grad_stats.items(), 
                                     key=lambda x: x[1]['norm'])[:5]
            for name, stats in sorted_grads_min:
                print(f"    {name}: norm={stats['norm']:.4e}")
            
            # Gradient flow plot
            GradientAnalyzer.plot_gradient_flow(
                model,
                step_dir / "gradient_flow.png"
            )
            
            # 3. Embedding visualizations
            print("\n3. EMBEDDING VISUALIZATIONS:")
            
            # Cosine similarity matrix
            EmbeddingVisualizer.plot_cosine_similarity_matrix(
                predictions,
                step_dir / "cosine_similarity_predictions.png"
            )
            EmbeddingVisualizer.plot_cosine_similarity_matrix(
                targets,
                step_dir / "cosine_similarity_targets.png"
            )
            try:
                EmbeddingVisualizer.plot_eigenvalue_spectrum(
                predictions,
                step_dir / "eigenvalues_predictions.png"
            )
            except Exception as e:
                print(f"⚠️ Eigen spectrum failed: {e}")

            # Eigenvalue spectrum 
            
            try:
                EmbeddingVisualizer.plot_eigenvalue_spectrum(
                targets,
                step_dir / "eigenvalues_targets.png"
            )
            except Exception as e:
                print(f"⚠️ Eigen spectrum failed: {e}")
            # PCA projection
            var_ratio = EmbeddingVisualizer.plot_pca_projection(
                predictions,
                save_path=step_dir / "pca_predictions.png"
            )
            print(f"  PCA explained variance (predictions): {var_ratio[0]:.3f}, {var_ratio[1]:.3f}")
            
            EmbeddingVisualizer.plot_pca_projection(
                targets,
                save_path=step_dir / "pca_targets.png"
            )
            
            # t-SNE projection
            EmbeddingVisualizer.plot_tsne_projection(
                predictions,
                save_path=step_dir / "tsne_predictions.png"
            )
            
            # Dimension usage
            EmbeddingVisualizer.plot_dimension_usage(
                predictions,
                save_path=step_dir / "dimension_usage_predictions.png"
            )
            EmbeddingVisualizer.plot_dimension_usage(
                targets,
                save_path=step_dir / "dimension_usage_targets.png"
            )
            
            # Correlation matrix
            EmbeddingVisualizer.plot_correlation_matrix(
                predictions,
                save_path=step_dir / "correlation_predictions.png"
            )
            
            print(f"  All visualizations saved to {step_dir}")
    
    def finalize(self):
        """Call at end of training to save final results"""
        self.metrics_logger.save()
        self.metrics_logger.plot_metrics(
            self.log_dir / "final_metrics.png"
        )
        print(f"\nAll diagnostics saved to {self.log_dir}")


# ==========================================================================
# Model Analyzer (analyze trained model)
# ==========================================================================
class ModelAnalyzer:
    """
    Analyze a trained model checkpoint.
    Run comprehensive diagnostics on saved model.
    """
    
    @staticmethod
    def analyze_checkpoint(model, dataloader, device='cuda', save_dir='./analysis'):
        """
        Comprehensive analysis of a trained model.
        
        Args:
            model: trained model
            dataloader: dataloader to extract embeddings from
            device: device to run on
            save_dir: where to save results
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print("="*60)
        print("MODEL ANALYSIS")
        print("="*60)
        
        model.eval()
        model.to(device)
        
        # Collect embeddings
        all_predictions = []
        all_targets = []
        all_input_ids = []
        
        print("\nCollecting embeddings...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 50:  # Limit for memory
                    break
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                predictions, targets, mask,context_embeddings  = model(input_ids, attention_mask)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_input_ids.append(input_ids.cpu())
        
        # Concatenate
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        print(f"Collected {all_predictions.shape[0]} samples")
        
        # Run all diagnostics
        print("\n" + "="*60)
        print("COLLAPSE ANALYSIS")
        print("="*60)
        
        collapse_pred = CollapseDetector.detect_collapse(all_predictions)
        collapse_target = CollapseDetector.detect_collapse(all_targets)
        
        print("\nPredictions:")
        for key, val in collapse_pred.items():
            print(f"  {key}: {val}")
        
        print("\nTargets:")
        for key, val in collapse_target.items():
            print(f"  {key}: {val}")
        
        # Visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        print("\n1. Cosine similarity matrices...")
        EmbeddingVisualizer.plot_cosine_similarity_matrix(
            all_predictions, save_path=save_dir / "cosine_sim_predictions.png"
        )
        EmbeddingVisualizer.plot_cosine_similarity_matrix(
            all_targets, save_path=save_dir / "cosine_sim_targets.png"
        )
        
        print("2. Eigenvalue spectra...")
        EmbeddingVisualizer.plot_eigenvalue_spectrum(
            all_predictions, save_path=save_dir / "eigenvalues_predictions.png"
        )
        EmbeddingVisualizer.plot_eigenvalue_spectrum(
            all_targets, save_path=save_dir / "eigenvalues_targets.png"
        )
        
        print("3. PCA projections...")
        EmbeddingVisualizer.plot_pca_projection(
            all_predictions, save_path=save_dir / "pca_predictions.png"
        )
        EmbeddingVisualizer.plot_pca_projection(
            all_targets, save_path=save_dir / "pca_targets.png"
        )
        
        print("4. t-SNE projections...")
        EmbeddingVisualizer.plot_tsne_projection(
            all_predictions, save_path=save_dir / "tsne_predictions.png"
        )
        EmbeddingVisualizer.plot_tsne_projection(
            all_targets, save_path=save_dir / "tsne_targets.png"
        )
        
        print("5. Dimension usage...")
        EmbeddingVisualizer.plot_dimension_usage(
            all_predictions, save_path=save_dir / "dim_usage_predictions.png"
        )
        EmbeddingVisualizer.plot_dimension_usage(
            all_targets, save_path=save_dir / "dim_usage_targets.png"
        )
        
        print("6. Correlation matrices...")
        EmbeddingVisualizer.plot_correlation_matrix(
            all_predictions, save_path=save_dir / "correlation_predictions.png"
        )
        EmbeddingVisualizer.plot_correlation_matrix(
            all_targets, save_path=save_dir / "correlation_targets.png"
        )
        
        print(f"\nAll analysis saved to {save_dir}")
        print("="*60)
        
        return {
            'collapse_predictions': collapse_pred,
            'collapse_targets': collapse_target,
        }


# ==========================================================================
# Example Integration with Training Loop
# ==========================================================================

def example_training_with_monitoring():
    """
    Example of how to integrate the monitoring into your training loop.
    """
    
    # Pseudo-code for integration
    print("""
Example integration into your training loop:

```python
from lejepa_debug_suite import TrainingMonitor, ModelAnalyzer

# Initialize monitor
monitor = TrainingMonitor(
    log_dir="./debug_logs",
    check_interval=100  # Run detailed diagnostics every 100 steps
)

# In your training loop:
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # ... your forward pass ...
        predictions, targets, mask = model(input_ids, attention_mask)
        
        # ... compute losses ...
        loss_dict = {
            'total_loss': loss.item(),
            'pred_loss': pred_loss.item(),
            'sigreg_loss': sigreg_loss.item(),
        }
        
        # ... backward pass ...
        loss.backward()
        
        # CHECK TRAINING HEALTH (before optimizer.step)
        monitor.check_training_health(
            model=model,
            predictions=predictions,
            targets=targets,
            loss_dict=loss_dict,
            batch_idx=batch_idx,
            epoch=epoch
        )
        
        # ... optimizer step ...
        optimizer.step()

# At end of training
monitor.finalize()

# Analyze final model
results = ModelAnalyzer.analyze_checkpoint(
    model=model,
    dataloader=val_loader,
    device='cuda',
    save_dir='./final_analysis'
)
```
""")


# ==========================================================================
# Quick diagnostic functions
# ==========================================================================

def quick_collapse_check(embeddings):
    """
    Quick one-liner to check if embeddings are collapsed.
    
    Usage:
        is_collapsed, info = quick_collapse_check(predictions)
        if is_collapsed:
            print("WARNING: Collapse detected!")
            print(info)
    """
    result = CollapseDetector.detect_collapse(embeddings)
    return result['collapse_detected'], result


def quick_gradient_check(model):
    """
    Quick one-liner to check gradient health.
    
    Usage:
        grad_norm = quick_gradient_check(model)
        if grad_norm < 1e-7:
            print("WARNING: Vanishing gradients!")
        elif grad_norm > 100:
            print("WARNING: Exploding gradients!")
    """
    return GradientAnalyzer.compute_gradient_norm(model)


def visualize_embeddings_all(embeddings, save_dir='./viz'):
    """
    Generate all visualizations for a set of embeddings.
    
    Usage:
        visualize_embeddings_all(predictions, save_dir='./my_viz')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print("Generating all visualizations...")
    
    EmbeddingVisualizer.plot_cosine_similarity_matrix(
        embeddings, save_path=save_dir / "cosine_sim.png"
    )
    EmbeddingVisualizer.plot_eigenvalue_spectrum(
        embeddings, save_path=save_dir / "eigenvalues.png"
    )
    EmbeddingVisualizer.plot_pca_projection(
        embeddings, save_path=save_dir / "pca.png"
    )
    EmbeddingVisualizer.plot_tsne_projection(
        embeddings, save_path=save_dir / "tsne.png"
    )
    EmbeddingVisualizer.plot_dimension_usage(
        embeddings, save_path=save_dir / "dim_usage.png"
    )
    EmbeddingVisualizer.plot_correlation_matrix(
        embeddings, save_path=save_dir / "correlation.png"
    )
    
    print(f"All visualizations saved to {save_dir}")


if __name__ == '__main__':
    print("LE-JEPA Debugging Suite")
    print("="*60)
    print()
    print("Available tools:")
    print("1. TrainingMonitor - comprehensive real-time monitoring")
    print("2. CollapseDetector - detect representation collapse")
    print("3. GradientAnalyzer - analyze gradient flow")
    print("4. EmbeddingVisualizer - visualize embeddings")
    print("5. ModelAnalyzer - analyze trained checkpoints")
    print()
    print("Quick functions:")
    print("  - quick_collapse_check(embeddings)")
    print("  - quick_gradient_check(model)")
    print("  - visualize_embeddings_all(embeddings)")
    print()
    example_training_with_monitoring() 