"""
Module 4: Denoising Autoencoder

1D convolutional autoencoder for signal denoising.
Reconstructs clean signals from noisy inputs.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import Tuple, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenoisingAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for signal denoising.
    Uses explicit upsampling to reliably reconstruct exact input dimensions.
    """
    
    def __init__(self, use_skip: bool = False):
        """
        Initialize autoencoder.
        
        Args:
            use_skip: Whether to use skip connections
        """
        super().__init__()
        self.use_skip = use_skip
        
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder (upsampling using interpolation + convolution)
        self.dec1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Conv1d(64, 12, kernel_size=5, padding=2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with explicit upsampling.
        
        Args:
            x: Input tensor (B, 12, T)
        
        Returns:
            Reconstructed signal of same shape
        """
        original_length = x.shape[-1]
        
        # Encoder
        e1 = self.enc1(x)  # (B, 32, T)
        e2_pool = nn.functional.max_pool1d(e1, kernel_size=2)  # (B, 32, T//2)
        e2 = self.enc2[1:](e2_pool)  # Apply conv after pooling
        
        e3_pool = nn.functional.max_pool1d(e2, kernel_size=2)  # (B, 64, T//4)
        e3 = self.enc3[1:](e3_pool)  # Apply conv after pooling
        
        # Bottleneck
        b = self.bottleneck(e3)  # (B, 32, T//4)
        
        # Decoder with explicit upsampling
        d1 = self.dec1(b)  # (B, 64, T//4)
        
        # Upsample to T//2
        d1_up = torch.nn.functional.interpolate(d1, scale_factor=2, mode='linear', align_corners=False)
        d2 = self.dec2(d1_up)  # (B, 64, T//2)
        
        # Upsample to T
        d2_up = torch.nn.functional.interpolate(d2, scale_factor=2, mode='linear', align_corners=False)
        d3 = self.dec3(d2_up)  # (B, 12, T)
        
        # Ensure exact match to original length
        if d3.shape[-1] != original_length:
            d3 = torch.nn.functional.interpolate(d3, size=original_length, mode='linear', align_corners=False)
        
        return d3


class DenoisingAutoEncoderTrainer:
    """Trainer for denoising autoencoder."""
    
    def __init__(self,
                 model: DenoisingAutoencoder,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3):
        """
        Initialize trainer.
        
        Args:
            model: Autoencoder model
            device: 'cpu' or 'cuda'
            learning_rate: Adam learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def prepare_batch(self, noisy_signals: list, 
                     clean_signals: list,
                     pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare variable-length signals for batch processing.
        
        Args:
            noisy_signals: List of (T, 12) arrays
            clean_signals: List of (T, 12) arrays
            pad_value: Value to use for padding
        
        Returns:
            Tuple of (noisy_batch, clean_batch, mask)
                noisy_batch: (B, 12, max_T)
                clean_batch: (B, 12, max_T)
                mask: (B, 1, max_T) boolean mask, True where data is valid
        """
        max_len = max([s.shape[0] for s in noisy_signals])
        
        noisy_batch = []
        clean_batch = []
        masks = []
        
        for noisy, clean in zip(noisy_signals, clean_signals):
            pad_len = max_len - noisy.shape[0]
            
            # Pad (add padding at the end)
            noisy_padded = np.pad(noisy, ((0, pad_len), (0, 0)), 
                                 mode='constant', constant_values=pad_value)
            clean_padded = np.pad(clean, ((0, pad_len), (0, 0)),
                                 mode='constant', constant_values=pad_value)
            
            # Create mask (True where data is valid, False where padded)
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
        """
        MSE loss with masking for padded regions.
        
        Args:
            output: Model output (B, 12, T)
            target: Target signal (B, 12, T)
            mask: Boolean mask (B, 1, T), True where valid
        
        Returns:
            Masked MSE loss
        """
        # Compute MSE per element
        mse = self.criterion(output, target)  # (B, 12, T)
        
        # Apply mask: expand mask to match MSE shape
        mask_expanded = mask.expand_as(mse)  # (B, 12, T)
        
        # Zero out masked regions
        masked_mse = mse * mask_expanded.float()
        
        # Compute mean only over valid regions
        loss = masked_mse.sum() / mask_expanded.float().sum()
        
        return loss
    
    def train_epoch(self, train_loader, batch_size: int = 16) -> float:
        """Train one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx in range(0, len(train_loader), batch_size):
            batch_noisy = train_loader[batch_idx:batch_idx + batch_size][0]
            batch_clean = train_loader[batch_idx:batch_idx + batch_size][1]
            
            noisy_batch, clean_batch, mask = self.prepare_batch(batch_noisy, batch_clean)
            
            self.optimizer.zero_grad()
            output = self.model(noisy_batch)
            loss = self.masked_mse_loss(output, clean_batch, mask)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def validate(self, val_loader, batch_size: int = 16) -> float:
        """Validate model."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx in range(0, len(val_loader), batch_size):
                batch_noisy = val_loader[batch_idx:batch_idx + batch_size][0]
                batch_clean = val_loader[batch_idx:batch_idx + batch_size][1]
                
                noisy_batch, clean_batch, mask = self.prepare_batch(batch_noisy, batch_clean)
                
                output = self.model(noisy_batch)
                loss = self.masked_mse_loss(output, clean_batch, mask)
                
                val_loss += loss.item()
                num_batches += 1
        
        avg_loss = val_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def fit(self, train_loader, val_loader, 
            epochs: int = 50, batch_size: int = 16, 
            patience: int = 7, checkpoint_dir: str = "checkpoints"):
        """
        Train model with early stopping.
        
        Args:
            train_loader: List of (noisy, clean) tuples
            val_loader: List of (noisy, clean) tuples
            epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            checkpoint_dir: Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("TRAINING DENOISING AUTOENCODER")
        logger.info("=" * 60)
        logger.info(f"Train samples: {len(train_loader)}")
        logger.info(f"Val samples: {len(val_loader)}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Patience: {patience}")
        logger.info("=" * 60)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, batch_size)
            val_loss = self.validate(val_loader, batch_size)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch + 1}/{epochs} | "
                       f"Train: {train_loss:.4f} | Val: {val_loss:.4f}", end="")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, 'dae_best.pt')
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f" ✓ (saved)")
            else:
                self.patience_counter += 1
                logger.info()
                
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            self.scheduler.step(val_loss)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)


def denoise(noisy_signal: np.ndarray,
           model_path: str = "checkpoints/dae_best.pt",
           device: str = 'cpu') -> np.ndarray:
    """
    Denoise a signal using the trained model.
    
    EXPORTED INTERFACE - downstream code uses this function.
    
    Args:
        noisy_signal: Input signal (T, 12) float32
        model_path: Path to saved model weights
        device: 'cpu' or 'cuda'
    
    Returns:
        Denoised signal (T, 12) float32
    """
    model = DenoisingAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Convert to tensor and reshape
    signal_tensor = torch.from_numpy(noisy_signal).float().to(device)
    signal_tensor = signal_tensor.transpose(0, 1).unsqueeze(0)  # (1, 12, T)
    
    with torch.no_grad():
        output = model(signal_tensor)
    
    # Convert back
    output_np = output.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
    
    return output_np
