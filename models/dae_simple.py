"""
Simplified Denoising Autoencoder - uses only regular Conv1D + upsampling.
Avoids transpose convolution dimension issues.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import Tuple
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenoisingAutoencoder(nn.Module):
    """
    Simple 1D Conv Autoencoder using only Conv1d + interpolate.
    Avoids transpose convolution dimension mismatches.
    """
    
    def __init__(self):
        super().__init__()
        
        # Encoder: downsampling via stride
        self.enc1 = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Bottleneck
        self.bottleneck = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        
        # Decoder: upsampling via interpolate + regular conv
        self.dec1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.dec3 = nn.Conv1d(64, 12, kernel_size=5, padding=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with explicit upsampling."""
        B, C, T = x.shape
        
        # Encoder
        e1 = self.enc1(x)  # (B, 32, T)
        e2 = self.enc2(e1)  # (B, 64, T/2)
        e3 = self.enc3(e2)  # (B, 64, T/4)
        
        # Bottleneck
        b = self.bottleneck(e3)  # (B, 32, T/4)
        
        # Decoder with explicit upsampling
        d1 = self.dec1(b)  # (B, 64, T/4)
        d1_up = torch.nn.functional.interpolate(d1, scale_factor=2, mode='linear', align_corners=False)  # (B, 64, T/2)
        
        d2 = self.dec2(d1_up)  # (B, 64, T/2)
        d2_up = torch.nn.functional.interpolate(d2, scale_factor=2, mode='linear', align_corners=False)  # (B, 64, T)
        
        d3 = self.dec3(d2_up)  # (B, 12, T)
        
        # Ensure exact original length
        if d3.shape[-1] != T:
            d3 = torch.nn.functional.interpolate(d3, size=T, mode='linear', align_corners=False)
        
        return d3


class DenoisingAutoEncoderTrainer:
    """Trainer for denoising autoencoder."""
    
    def __init__(self,
                 model: DenoisingAutoencoder,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3):
        """Initialize trainer."""
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    @staticmethod
    def _to_time_channel(signal) -> np.ndarray:
        """Convert a signal to float32 ndarray with shape (T, 12)."""
        if isinstance(signal, torch.Tensor):
            arr = signal.detach().cpu().numpy()
        else:
            arr = np.asarray(signal)

        # Remove singleton dimensions from collate artifacts.
        arr = np.squeeze(arr)

        if arr.ndim == 0:
            arr = np.zeros((1, 12), dtype=np.float32)
        elif arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim > 2:
            # Flatten all non-time dims into channels.
            arr = arr.reshape(arr.shape[0], -1)

        # Heuristic: if channels are likely first dim, transpose.
        if arr.ndim == 2 and arr.shape[0] == 12 and arr.shape[1] != 12:
            arr = arr.T

        # Ensure exactly 12 channels.
        if arr.shape[1] < 12:
            arr = np.pad(arr, ((0, 0), (0, 12 - arr.shape[1])), mode='constant')
        elif arr.shape[1] > 12:
            arr = arr[:, :12]

        return arr.astype(np.float32, copy=False)
    
    def prepare_batch(self, noisy_signals: list, 
                     clean_signals: list,
                     pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare variable-length signals for batch processing."""
        if isinstance(noisy_signals, np.ndarray) and noisy_signals.ndim == 2:
            noisy_signals = [noisy_signals]
        if isinstance(clean_signals, np.ndarray) and clean_signals.ndim == 2:
            clean_signals = [clean_signals]

        noisy_proc = [self._to_time_channel(s) for s in noisy_signals]
        clean_proc = [self._to_time_channel(s) for s in clean_signals]
        max_len = max([s.shape[0] for s in noisy_proc])
        
        noisy_batch = []
        clean_batch = []
        masks = []
        
        for noisy, clean in zip(noisy_proc, clean_proc):
            pad_len = max_len - noisy.shape[0]
            
            # Pad at the end
            noisy_padded = np.pad(noisy, ((0, pad_len), (0, 0)), 
                                 mode='constant', constant_values=pad_value)
            clean_padded = np.pad(clean, ((0, pad_len), (0, 0)),
                                 mode='constant', constant_values=pad_value)
            
            # Create mask (True where valid data, False where padded)
            mask = np.ones(max_len, dtype=bool)
            mask[noisy.shape[0]:] = False
            
            noisy_batch.append(noisy_padded)
            clean_batch.append(clean_padded)
            masks.append(mask)
        
        # Convert to tensors: (B, T, 12) -> (B, 12, T)
        noisy_batch = torch.from_numpy(np.array(noisy_batch)).transpose(1, 2).float().to(self.device)
        clean_batch = torch.from_numpy(np.array(clean_batch)).transpose(1, 2).float().to(self.device)
        mask = torch.from_numpy(np.array(masks)).unsqueeze(1).to(self.device)  # (B, 1, T)
        
        return noisy_batch, clean_batch, mask
    
    def masked_mse_loss(self, output: torch.Tensor,
                       target: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        """MSE loss with masking for padded regions."""
        # Ensure shapes match
        if output.shape != target.shape:
            target = torch.nn.functional.interpolate(target, size=output.shape[-1], mode='linear', align_corners=False)
        
        # Compute MSE per element
        mse = self.criterion(output, target)  # (B, 12, T)
        
        # Apply mask
        mask_expanded = mask.expand_as(mse)  # (B, 12, T)
        masked_mse = mse * mask_expanded.float()
        
        # Compute mean only over valid regions
        loss = masked_mse.sum() / mask_expanded.float().sum().clamp(min=1e-6)
        
        return loss
    
    def train_epoch(self, train_loader, batch_size: int = 16) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch_start in range(0, len(train_loader), batch_size):
            batch = train_loader[batch_start:batch_start + batch_size]
            noisy_signals = [sample[0] for sample in batch]
            clean_signals = [sample[1] for sample in batch]

            noisy_batch, clean_batch, mask = self.prepare_batch(noisy_signals, clean_signals)
            
            # Forward pass
            output = self.model(noisy_batch)
            
            # Compute loss
            loss = self.masked_mse_loss(output, clean_batch, mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / max(len(train_loader), 1)
        return avg_loss
    
    def val_epoch(self, val_loader) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_start in range(0, len(val_loader), 16):
                batch = val_loader[batch_start:batch_start + 16]
                noisy_signals = [sample[0] for sample in batch]
                clean_signals = [sample[1] for sample in batch]

                noisy_batch, clean_batch, mask = self.prepare_batch(noisy_signals, clean_signals)
                
                # Forward pass
                output = self.model(noisy_batch)
                
                # Compute loss
                loss = self.masked_mse_loss(output, clean_batch, mask)
                total_loss += loss.item()
        
        avg_loss = total_loss / max(len(val_loader), 1)
        return avg_loss
    
    def fit(self, train_loader, val_loader, 
            epochs: int = 50, batch_size: int = 16, 
            checkpoint_path: str = 'checkpoints/dae_best.pt'):
        """Train the model."""
        logger.info("=" * 60)
        logger.info("TRAINING DENOISING AUTOENCODER")
        logger.info("=" * 60)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Patience: 7")
        logger.info("=" * 60)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, batch_size)
            val_loss = self.val_epoch(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | ✓ (saved)")
            else:
                self.patience_counter += 1
                logger.info(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Patience: {self.patience_counter}/7")
                
                if self.patience_counter >= 7:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(checkpoint_path))
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        
        return self.train_losses, self.val_losses
