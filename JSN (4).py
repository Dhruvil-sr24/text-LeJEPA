"""
LE-JEPA Training Script with Complete Debugging Integration

This script integrates all debugging tools into your training loop,
giving you complete visibility into what's happening.

Run this to train with full monitoring and diagnostics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm
from torch import amp

import sys

# Import your LE-JEPA model (from the previous artifact)
from main import TextLeJEPA, SIGRegLoss, TextDataset, prepare_dataloaders

# Import debugging suite
from lejepa_debug_suite import TrainingMonitor, ModelAnalyzer, quick_collapse_check


# ==========================================================================
# Debuggable Trainer (Enhanced with full monitoring)
# ==========================================================================
class DebuggableLeJEPATrainer:
    """
    Enhanced trainer with comprehensive debugging and monitoring.
    
    Features:
    - Real-time collapse detection
    - Gradient flow visualization
    - Embedding quality tracking
    - Automatic problem detection and warnings
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        learning_rate=3e-4,
        weight_decay=0.05,
        sigreg_weight=1.0,
        warmup_steps=500,
        max_steps=None,
        device="cuda",
        use_amp=True,
        # Debugging parameters
        enable_monitoring=True,
        monitor_interval=100,
        log_dir="./debug_logs",
    ):
        self.model = model.to(device)
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.use_amp = use_amp
        self.sigreg_weight=sigreg_weight
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
            eps=1e-8
        )
        # self.sigreg_lambda0 = sigreg_weight  # e.g. 1.0
        # self.sigreg_tau = 300
        self.sigreg_lambda0 = 0.1
        self.lambda_min = 0.01
        self.lambda_max = 0.5
        self.rank_target = 0.45
        self.kp = 0.65

        # Scheduler
        total_steps = max_steps or (len(train_dataloader) * 10)
        self.scheduler = self._get_cosine_schedule(
            self.optimizer, warmup_steps, total_steps
        )
        
        # Losses
        from lejepa_debug_suite import quick_collapse_check  # You need to import this
        # SIGRegLoss should be imported from your model file
        self.sigreg = SIGRegLoss(
                    num_sketches=16,
                    sketch_dim=128,
                    reg_weight=self.sigreg_weight,
                    max_samples=2048
                ).to(self.device)  # Initialize with actual SIGRegLoss
        self.prediction_loss = nn.MSELoss()
        
        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if (
            use_amp and device == "cuda"
        ) else None
        
        # Monitoring
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            from lejepa_debug_suite import TrainingMonitor
            self.monitor = TrainingMonitor(
                log_dir=log_dir,
                check_interval=monitor_interval
            )
        
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Problem tracking
        self.collapse_warnings = 0
        self.grad_warnings = 0
    
    # def sigreg_weight_at_step(self):

    #     return self.sigreg_lambda0 * math.exp(
    #         -self.global_step / self.sigreg_tau
    #     )

    def adaptive_sigreg_weight(self, rank_ratio):
        error = self.rank_target - rank_ratio
        lambda_t = self.sigreg_lambda0 * (0.5 + self.kp * error)
        return float(torch.clamp(
            torch.tensor(lambda_t),
            self.lambda_min,
            self.lambda_max
        ))

    def _get_cosine_schedule(self, optimizer, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _detect_problems(self, predictions, targets, grad_norm, loss_dict):
        """
        Detect training problems and issue warnings.
        """
        from lejepa_debug_suite import quick_collapse_check
        
        problems = []
        
        # Check for collapse
        is_collapsed, info = quick_collapse_check(predictions)
        if is_collapsed:
            self.collapse_warnings += 1
            problems.append(f"⚠️  COLLAPSE DETECTED (warning #{self.collapse_warnings})")
            problems.append(f"   Effective rank: {info['effective_rank']:.2f}")
            problems.append(f"   Cosine sim: {info['cos_sim_mean']:.4f}")
        
        # Check gradients
        if grad_norm < 1e-7:
            self.grad_warnings += 1
            problems.append(f"⚠️  VANISHING GRADIENTS (warning #{self.grad_warnings})")
            problems.append(f"   Gradient norm: {grad_norm:.2e}")
        elif grad_norm > 100:
            self.grad_warnings += 1
            problems.append(f"⚠️  EXPLODING GRADIENTS (warning #{self.grad_warnings})")
            problems.append(f"   Gradient norm: {grad_norm:.2e}")
        
        # Check loss
        if torch.isnan(loss_dict['total_loss']) or torch.isinf(loss_dict['total_loss']):
    
            problems.append(f"⚠️  NaN/Inf LOSS DETECTED")
        
        # Check SIGReg
        if loss_dict.get('sigreg_loss', 0) > 10 * loss_dict.get('pred_loss', 1):
            problems.append(f"⚠️  SIGREG DOMINATING - might need to reduce weight")
        
        if problems:
            print("\n" + "="*60)
            print("TRAINING PROBLEMS DETECTED:")
            for p in problems:
                print(p)
            print("="*60 + "\n")
        
        return len(problems) > 0
    
    def train_step(self, batch, epoch, batch_idx):
        """Single training step with full debugging"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        

        use_amp = self.scaler is not None

        with amp.autocast(device_type="cuda", enabled=use_amp):

            # Forward pass
            predictions, targets, mask, context_embeddings = self.model(input_ids, attention_mask)
            
            # Prediction loss (only masked tokens)
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

            enc_valid = context_embeddings[
                valid[:, :context_embeddings.size(1)].expand_as(context_embeddings)
            ].reshape(-1, context_embeddings.size(-1))

            sigreg_loss = self.sigreg(enc_valid)

            # valid = attention_mask.bool().unsqueeze(-1)
            # pred_valid = predictions[valid.expand_as(predictions)].reshape(
            #     -1, predictions.size(-1)
            # )
            
            # sigreg_loss = self.sigreg(pred_valid) if self.sigreg else torch.tensor(0.0)
            
            # Total loss
            # current_lambda = self.sigreg_weight_at_step()
            from lejepa_debug_suite import CollapseDetector
            with torch.no_grad():
                collapse_info = CollapseDetector.detect_collapse(enc_valid)
                rank_ratio = collapse_info["rank_ratio"]

            current_lambda = self.adaptive_sigreg_weight(
                rank_ratio
            )
            loss = pred_loss + current_lambda * sigreg_loss

            # loss = pred_loss + self.sigreg_weight*sigreg_loss
        
        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.scheduler.step()
        
        # Prepare metrics
        loss_dict = {
            'total_loss': loss,            # 🔴 TENSOR
            'pred_loss': pred_loss,        # 🔴 TENSOR
            'sigreg_loss': sigreg_loss,    # 🔴 TENSOR
            'learning_rate': self.scheduler.get_last_lr()[0],
        }

        # MONITORING & DEBUGGING
        if self.enable_monitoring:
            self.monitor.check_training_health(
                model=self.model,
                predictions=predictions,
                targets=targets,
                loss_dict=loss_dict,
                batch_idx=batch_idx,
                epoch=epoch
            )
        
        # Problem detection
        self._detect_problems(predictions, targets, grad_norm, loss_dict)
        
        self.global_step += 1
        
        return loss_dict
    
    def train_epoch(self, epoch):
        """Train one epoch with monitoring"""
        self.model.train()
        total_metrics = {
            'loss': 0.0,
            'pred_loss': 0.0,
            'sigreg_loss': 0.0
        }
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch, epoch, batch_idx)
            
            for k in total_metrics:
                if k in metrics:
                    total_metrics[k] += metrics.get(k, 0)
            
            pbar.set_postfix({
                    "loss": f"{metrics['total_loss'].item():.4f}",
                    "pred": f"{metrics['pred_loss'].item():.4f}",
                    "sig": f"{metrics['sigreg_loss'].item():.4f}",
                    "lr": f"{metrics['learning_rate']:.6f}",
                })

        
        n = len(self.train_dataloader)
        return {k: v / n for k, v in total_metrics.items()}
    
    @torch.no_grad()
    def validate(self):
        """Validation with diagnostics"""
        self.model.eval()
        total_metrics = {
            'loss': 0.0,
            'pred_loss': 0.0,
            'sigreg_loss': 0.0
        }
        
        # Collect some embeddings for analysis
        val_predictions = []
        val_targets = []
        
        for batch in tqdm(self.val_dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            predictions, targets, mask, context_embeddings  = self.model(input_ids, attention_mask)
            
            # Store for analysis
            if len(val_predictions) < 10:  # Collect first 10 batches
                val_predictions.append(predictions.cpu())
                val_targets.append(targets.cpu())
            
            # Compute loss
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
            
            # valid = attention_mask.bool().unsqueeze(-1)
            # pred_valid = predictions[valid.expand_as(predictions)].reshape(
            #     -1, predictions.size(-1)
            # )
            # sigreg_loss = self.sigreg(pred_valid) if self.sigreg else torch.tensor(0.0)
            valid = attention_mask.bool().unsqueeze(-1)

            enc_valid = context_embeddings[
                valid[:, :context_embeddings.size(1)].expand_as(context_embeddings)
            ].reshape(-1, context_embeddings.size(-1))

            sigreg_loss = self.sigreg(enc_valid)

            loss = pred_loss
            
            # total_metrics['loss'] += loss.item()
            # total_metrics['pred_loss'] += pred_loss.item()
            # total_metrics['sigreg_loss'] += sigreg_loss.item() if isinstance(sigreg_loss, torch.Tensor) else sigreg_loss
            total_metrics['loss'] += loss.item()
            total_metrics['pred_loss'] += pred_loss.item()
            total_metrics['sigreg_loss'] += (
                sigreg_loss.item() if isinstance(sigreg_loss, torch.Tensor) else sigreg_loss
            )


        # Analyze validation embeddings
        if val_predictions:
            from lejepa_debug_suite import quick_collapse_check
            val_pred_all = torch.cat(val_predictions, dim=0)
            is_collapsed, info = quick_collapse_check(val_pred_all)
            
            print(f"\nValidation embedding analysis:")
            print(f"  Effective rank: {info['effective_rank']:.2f}")
            print(f"  Rank ratio: {info['rank_ratio']:.3f}")
            print(f"  Cosine similarity: {info['cos_sim_mean']:.4f}")
            print(f"  Collapsed: {is_collapsed}")
        
        n = len(self.val_dataloader)
        return {k: v / n for k, v in total_metrics.items()}
    
    def train(self, num_epochs, save_dir="./checkpoints"):
        """Full training with monitoring"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("="*60)
        print("TRAINING WITH FULL DEBUGGING")
        print("="*60)
        print(f"Monitoring enabled: {self.enable_monitoring}")
        print(f"Log directory: {self.monitor.log_dir if self.enable_monitoring else 'N/A'}")
        print(f"Epochs: {num_epochs}")
        print("="*60 + "\n")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Collapse warnings so far: {self.collapse_warnings}")
            print(f"  Gradient warnings so far: {self.grad_warnings}")
            
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
                print(f"  ✓ Saved best model to {save_path}")
        
        # Finalize monitoring
        if self.enable_monitoring:
            print("\nFinalizing debugging logs...")
            self.monitor.finalize()
        
        # Final analysis
        print("\n" + "="*60)
        print("FINAL MODEL ANALYSIS")
        print("="*60)
        
        from lejepa_debug_suite import ModelAnalyzer
        final_results = ModelAnalyzer.analyze_checkpoint(
            model=self.model,
            dataloader=self.val_dataloader,
            device=self.device,
            save_dir='./final_analysis'
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Total collapse warnings: {self.collapse_warnings}")
        print(f"Total gradient warnings: {self.grad_warnings}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)


# ==========================================================================
# Main script with example usage
# ==========================================================================

def main():
    """
    Main training script with full debugging.
    
    This shows you exactly how to use all the debugging tools.
    """
    
    print("""
    =============================================
    LE-JEPA TRAINING WITH COMPLETE DEBUGGING
    =============================================
    
    This script will:
    1. Monitor training in real-time
    2. Detect collapse automatically
    3. Track gradient health
    4. Generate visualizations every N steps
    5. Provide comprehensive final analysis
    
    All results saved to:
    - ./debug_logs/        (training monitoring)
    - ./checkpoints/       (model checkpoints)
    - ./final_analysis/    (final model analysis)
    
    =============================================
    """)
    
    # Configuration
    CONFIG = {
        'vocab_size': 50000,
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'max_seq_len': 128,
        'predictor_depth': 2,
        'mask_ratio': 0.3,
        
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 0.05,
        'sigreg_weight': 2.0,
        'num_epochs': 5,  # Small for testing
        'warmup_steps': 400,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_amp': True,
        
        'dataset_size': 50000,  # Small for testing
        'num_workers': 4,
        
        # Debugging settings
        'enable_monitoring': True,
        'monitor_interval': 50,  # Run diagnostics every 50 steps
    }
    
    print("Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print()
    
    # NOTE: You need to import your actual model classes here
    # This is pseudo-code showing the structure
    
    print("⚠️  To run this script, you need to:")
    print("1. Save the LE-JEPA model from previous artifact as 'text_lejepa.py'")
    print("2. Save the debugging suite as 'lejepa_debug_suite.py'")
    print("3. Import them at the top of this file")
    print("4. Then run this script!")
    print()
    print("Example imports needed:")
    print("  from text_lejepa import TextLeJEPA, SIGRegLoss, prepare_dataloaders")
    print("  from lejepa_debug_suite import TrainingMonitor, ModelAnalyzer")
    
    # Uncomment when you have the imports:
    
    # Prepare data
    train_loader, val_loader, tokenizer = prepare_dataloaders(
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        dataset_size=CONFIG['dataset_size'],
        seq_len=CONFIG['max_seq_len']
    )
    
    # Create model
    model = TextLeJEPA(
        vocab_size=CONFIG['vocab_size'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        max_seq_len=CONFIG['max_seq_len'],
        predictor_depth=CONFIG['predictor_depth'],
        mask_ratio=CONFIG['mask_ratio']
    )
    
    # Create debuggable trainer
    trainer = DebuggableLeJEPATrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        sigreg_weight=CONFIG['sigreg_weight'],
        warmup_steps=CONFIG['warmup_steps'],
        device=CONFIG['device'],
        use_amp=CONFIG['use_amp'],
        enable_monitoring=CONFIG['enable_monitoring'],
        monitor_interval=CONFIG['monitor_interval'],
    )
    
    # Initialize SIGReg
    # trainer.sigreg = SIGRegLoss(
    #     num_sketches=3,
    #     sketch_dim=128,
    #     reg_weight=CONFIG['sigreg_weight'],
    #     max_samples=2048
    # ).to(CONFIG['device'])
    
    # Train!
    trainer.train(num_epochs=CONFIG['num_epochs'])


if __name__ == '__main__':
    main()