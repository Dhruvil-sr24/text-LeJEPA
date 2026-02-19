"""
Text LE-JEPA: Latent-Euclidean JEPA for Text

LE-JEPA's Anti-Collapse Mechanism (from the paper):
1. Prediction task: Standard JEPA objective (predict masked tokens)
2. SIGReg: Enforce isotropic Gaussian distribution on embeddings
   - Prevents collapse by construction (provably correct)
   - No need for EMA, stop-gradient, or other heuristics
   - Scales linearly with dimension and batch size

Key insight: Isotropic Gaussian embeddings minimize worst-case risk
across downstream tasks AND prevent representation collapse.

Reference: "Latent-Euclidean JEPA" (LeJEPA), Balestriero & LeCun, 2025
Author: LE-JEPA implementation for text
Date: 2026-01-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


# ==========================================================================
# SIGReg: Sketched Isotropic Gaussian Regularization (Core of LE-JEPA)
# ==========================================================================
class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg)
    
    Forces embeddings to follow an isotropic Gaussian distribution N(0, I)
    using random projections and characteristic function matching.
    
    Key properties:
    - Provably prevents collapse
    - Linear complexity in dimension and batch size
    - No hyperparameter tuning needed
    - Works without EMA or stop-gradient tricks
    
    Args:
        num_sketches: Number of random projections (2-4 is sufficient)
        sketch_dim: Dimension of projected space (64-128 works well)
        reg_weight: Weight of regularization term (0.01-0.1 typical)
        max_samples: Subsample tokens if batch too large (memory efficiency)
    """
    def __init__(
        self,
        num_sketches=3,
        sketch_dim=128,
        reg_weight=0.05,
        max_samples=2048
    ):
        super().__init__()
        self.num_sketches = num_sketches
        self.sketch_dim = sketch_dim
        # self.reg_weight = reg_weight
        self.max_samples = max_samples
    
    def epps_pulley_statistic(self, x, y):
        """
        Epps-Pulley two-sample test statistic.
        Measures distributional difference via pairwise distances.
        
        Args:
            x, y: [N] 1D tensors (projected embeddings)
        Returns:
            Scalar test statistic
        """
        n, m = x.shape[0], y.shape[0]
        if n == 0 or m == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Compute pairwise distances (use float32 for stability)
        x_f32 = x.unsqueeze(-1).float()
        y_f32 = y.unsqueeze(-1).float()
        
        # Within-distribution distances
        xx = torch.cdist(x_f32, x_f32, p=2).sum() / (n * n)
        yy = torch.cdist(y_f32, y_f32, p=2).sum() / (m * m)
        
        # Cross-distribution distances
        xy = torch.cdist(x_f32, y_f32, p=2).sum() / (n * m)
        
        # EP statistic: 2*E[d(X,Y)] - E[d(X,X')] - E[d(Y,Y')]
        stat = 2.0 * xy - xx - yy
        
        return stat.to(x.dtype)
    
    def forward(self, embeddings):
        """
        Compute SIGReg loss.
        
        Args:
            embeddings: [B, L, D] or [N, D] tensor of embeddings
        Returns:
            Scalar regularization loss
        """
        # Flatten to [N, D] if needed
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        N, D = embeddings.shape
        
        # Subsample if too many tokens (memory efficiency)
        if N > self.max_samples:
            idx = torch.randperm(N, device=embeddings.device)[:self.max_samples]
            embeddings = embeddings[idx]
            N = self.max_samples
        
        # Sample reference from standard normal
        reference = torch.randn_like(embeddings)
        
        total_loss = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
        
        # Multiple random projections (sketches)
        for _ in range(self.num_sketches):
            # Random projection matrix [D, sketch_dim]
            # Normalized by sqrt(D) for stable gradients
            projection = torch.randn(
                D, self.sketch_dim,
                device=embeddings.device,
                dtype=embeddings.dtype
            ) / math.sqrt(D)
            
            # Project both distributions: [N, sketch_dim]
            emb_proj = embeddings @ projection
            ref_proj = reference @ projection
            
            # Compute EP statistic for each projected dimension
            for d in range(self.sketch_dim):
                ep_stat = self.epps_pulley_statistic(
                    emb_proj[:, d],
                    ref_proj[:, d]
                )
                # Square the statistic (as in paper)
                total_loss = total_loss + ep_stat * ep_stat
        
        # Average over sketches and dimensions
        reg_loss = total_loss / (self.num_sketches * self.sketch_dim)
        
        return  reg_loss


# ==========================================================================
# Text LE-JEPA Architecture
# ==========================================================================
class TextLeJEPA(nn.Module):
    """
    Text LE-JEPA: Simple JEPA with SIGReg for collapse prevention.
    
    Key architectural choices:
    - Single encoder (no separate context/target - simpler than I-JEPA)
    - Predictor operates on masked sequence
    - NO EMA, NO stop-gradient - SIGReg handles collapse
    - Targets computed with same encoder under no_grad for efficiency
    """
    def __init__(
        self,
        vocab_size=30522,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        max_seq_len=128,
        predictor_depth=1,
        mask_ratio=0.5,
        dropout=0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Main encoder (processes visible tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Predictor (predicts masked tokens from visible context)
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            predictor_layer,
            num_layers=predictor_depth
        )
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Projection heads (simple linear layers)
        self.predictor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.target_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following best practices"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def random_masking(self, x, mask_ratio):
        """
        Random masking strategy.
        
        Args:
            x: [B, L, D] input embeddings
            mask_ratio: fraction of tokens to mask
        Returns:
            x_masked: [B, L_keep, D] visible tokens
            mask: [B, L] binary mask (1 = masked)
            ids_restore: [B, L] indices to restore original order
        """
        batch_size, seq_len, dim = x.shape
        len_keep = max(1, int(seq_len * (1 - mask_ratio)))
        
        # Random shuffle
        noise = torch.rand(batch_size, seq_len, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, dim)
        )
        
        # Create mask: 0 = keep, 1 = remove
        mask = torch.ones(batch_size, seq_len, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, input_ids, attention_mask=None, return_embeddings=False):
        """
        Forward pass.
        
        Args:
            input_ids: [B, L] token indices
            attention_mask: [B, L] attention mask
            return_embeddings: if True, return embeddings for downstream tasks
        Returns:
            predictions: [B, L, D] predicted embeddings
            targets: [B, L, D] target embeddings
            mask: [B, L] binary mask
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.embedding_dropout(x)
        
        # For downstream tasks, return full embeddings
        if return_embeddings:
            return self.encoder(x)
        
        # Apply random masking
        x_visible, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # Encode visible tokens
        context_embeddings = self.encoder(x_visible)
        
        # Reconstruct full sequence with mask tokens
        batch_size, len_keep, dim = context_embeddings.shape
        x_full = torch.zeros(
            batch_size, seq_len, dim,
            device=x.device, dtype=x.dtype
        )
        
        # Scatter visible embeddings back to original positions
        ids_keep = torch.argsort(ids_restore, dim=1)[:, :len_keep]
        x_full.scatter_(
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, dim),
            src=context_embeddings
        )
        
        # Fill masked positions with learnable mask tokens
        mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, dim)
        x_full = x_full * (1 - mask_expanded) + mask_tokens * mask_expanded
        
        # Predict masked tokens
        predictions = self.predictor(x_full)
        predictions = self.predictor_proj(predictions)
        
        # Compute targets (no gradient - efficiency only)
        with torch.no_grad():
            targets = self.encoder(x)
            # targets = self.target_proj(targets)
        
        return predictions, targets, mask, context_embeddings


# ==========================================================================
# Dataset & DataLoader
# ==========================================================================
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }


def prepare_dataloaders(
    batch_size=64,
    num_workers=4,
    dataset_size=50000,
    seq_len=128
):
    """Prepare train/val dataloaders"""
    print("Loading TinyStories dataset...")
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=f"train[:{dataset_size}]"
    )
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    train_dataset = TextDataset(
        dataset["train"]["text"],
        tokenizer,
        max_length=seq_len
    )
    val_dataset = TextDataset(
        dataset["test"]["text"],
        tokenizer,
        max_length=seq_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer


# ==========================================================================
# LE-JEPA Trainer
# ==========================================================================
class TextLeJEPATrainer:
    """
    LE-JEPA trainer with SIGReg regularization.
    
    Key points:
    - Joint optimization of prediction + SIGReg
    - NO EMA, NO stop-gradient tricks needed
    - Simple cosine learning rate schedule
    - Gradient clipping for stability
    """
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        learning_rate=3e-4,
        weight_decay=0.05,
        sigreg_weight=0.05,
        warmup_steps=500,
        max_steps=None,
        device="cuda",
        use_amp=True,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.use_amp = use_amp
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
            eps=1e-8
        )
        
        # Learning rate scheduler
        total_steps = max_steps or (len(train_dataloader) * 10)
        self.scheduler = self._get_cosine_schedule(
            self.optimizer,
            warmup_steps,
            total_steps
        )
        
        # LE-JEPA losses
        self.sigreg = SIGRegLoss(
            num_sketches=3,
            sketch_dim=128,
            reg_weight=sigreg_weight,
            max_samples=2048
        ).to(device)
        self.prediction_loss = nn.MSELoss()
        
        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if (
            use_amp and device == "cuda"
        ) else None
        
        self.global_step = 0
        self.best_val_loss = float("inf")
    
    def _get_cosine_schedule(self, optimizer, warmup_steps, total_steps):
        """Cosine learning rate schedule with warmup"""
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_step(self, batch):
        """Single training step"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        with torch.amp.autocast("cuda", enabled=(self.scaler is not None)):
            # Forward pass
            predictions, targets, mask = self.model(input_ids, attention_mask)
            
            # Prediction loss (only on masked tokens)
            mask_expanded = mask.unsqueeze(-1).expand_as(predictions).bool()
            pred_masked = predictions[mask_expanded].reshape(
                -1, predictions.size(-1)
            )
            target_masked = targets[mask_expanded].reshape(
                -1, targets.size(-1)
            )
            
            if pred_masked.numel() > 0:
                pred_loss = self.prediction_loss(pred_masked, target_masked)
            else:
                pred_loss = torch.tensor(0.0, device=self.device)
            
            # SIGReg loss (on all valid predictions)
            # Filter by attention mask to only use real tokens
            valid = attention_mask.bool().unsqueeze(-1)
            pred_valid = predictions[valid.expand_as(predictions)].reshape(
                -1, predictions.size(-1)
            )
            
            sigreg_loss = self.sigreg(pred_valid)
            
            # Total loss
            loss = pred_loss + sigreg_loss
        
        # Optimization step
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.scheduler.step()
        self.global_step += 1
        
        return {
            "loss": loss.item(),
            "pred_loss": pred_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_metrics = {
            'loss': 0.0,
            'pred_loss': 0.0,
            'sigreg_loss': 0.0
        }
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            metrics = self.train_step(batch)
            
            for k in total_metrics:
                total_metrics[k] += metrics[k]
            
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "pred": f"{metrics['pred_loss']:.4f}",
                "sig": f"{metrics['sigreg_loss']:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
            })
        
        n = len(self.train_dataloader)
        return {k: v / n for k, v in total_metrics.items()}
    
    @torch.no_grad()
    def validate(self):
        """Validation pass"""
        self.model.eval()
        total_metrics = {
            'loss': 0.0,
            'pred_loss': 0.0,
            'sigreg_loss': 0.0
        }
        
        for batch in tqdm(self.val_dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            predictions, targets, mask = self.model(input_ids, attention_mask)
            
            # Prediction loss
            mask_expanded = mask.unsqueeze(-1).expand_as(predictions).bool()
            pred_masked = predictions[mask_expanded].reshape(
                -1, predictions.size(-1)
            )
            target_masked = targets[mask_expanded].reshape(
                -1, targets.size(-1)
            )
            
            if pred_masked.numel() > 0:
                pred_loss = self.prediction_loss(pred_masked, target_masked)
            else:
                pred_loss = torch.tensor(0.0, device=self.device)
            
            # SIGReg loss
            valid = attention_mask.bool().unsqueeze(-1)
            pred_valid = predictions[valid.expand_as(predictions)].reshape(
                -1, predictions.size(-1)
            )
            sigreg_loss = self.sigreg(pred_valid)
            
            # LE-JEPA validation uses only prediction loss
            loss = pred_loss
            
            total_metrics['loss'] += loss.item()
            total_metrics['pred_loss'] += pred_loss.item()
            total_metrics['sigreg_loss'] += sigreg_loss.item()
        
        n = len(self.val_dataloader)
        return {k: v / n for k, v in total_metrics.items()}
    
    def train(self, num_epochs, save_dir="./checkpoints"):
        """Full training loop"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting LE-JEPA training for {num_epochs} epochs...")
        print("Anti-collapse mechanism: SIGReg (no EMA, no heuristics)\n")
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f} | "
                  f"Pred: {train_metrics['pred_loss']:.4f} | "
                  f"SIGReg: {train_metrics['sigreg_loss']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                save_path = os.path.join(save_dir, 'text_lejepa_best.pt')
                torch.save({
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                }, save_path)
                print(f"✓ Saved best model to {save_path}\n")


# ==========================================================================
# Main Training Script
# ==========================================================================
CONFIG = {
    # Model architecture
    'vocab_size': 30522,
    'hidden_dim': 512,
    'num_layers': 4,
    'num_heads': 8,
    'max_seq_len': 128,
    'predictor_depth': 2,
    'mask_ratio': 0.5,
    
    # Training
    'batch_size': 64,
    'learning_rate': 3e-4,
    'weight_decay': 0.05,
    'sigreg_weight': 0.05,  # Key parameter - adjust if needed
    'num_epochs': 20,
    'warmup_steps': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_amp': True,
    
    # Data
    'dataset_size': 50000,
    'num_workers': 4,
}


def main():
    cfg = CONFIG
    print("=" * 60)
    print("Text LE-JEPA Configuration")
    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print()
    
    # Prepare data
    train_loader, val_loader, tokenizer = prepare_dataloaders(
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        dataset_size=cfg['dataset_size'],
        seq_len=cfg['max_seq_len']
    )
    
    # Create model
    model = TextLeJEPA(
        vocab_size=cfg['vocab_size'],
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        max_seq_len=cfg['max_seq_len'],
        predictor_depth=cfg['predictor_depth'],
        mask_ratio=cfg['mask_ratio']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}\n")
    
    # Create trainer
    trainer = TextLeJEPATrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
        sigreg_weight=cfg['sigreg_weight'],
        warmup_steps=cfg['warmup_steps'],
        device=cfg['device'],
        use_amp=cfg['use_amp']
    )
    
    # Train
    trainer.train(num_epochs=cfg['num_epochs'])
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()